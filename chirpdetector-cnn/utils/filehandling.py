import copy
import os
import pathlib

import nixio as nio
import numpy as np
import yaml
from IPython import embed
from thunderfish.dataloader import DataLoader


class ConfLoader:
    """
    Load configuration from yaml file as class attributes
    """

    def __init__(self, path: str) -> None:
        with open(path) as file:
            try:
                conf = yaml.safe_load(file)
                for key in conf:
                    setattr(self, key, conf[key])
            except yaml.YAMLError as error:
                raise error


class NumpyLoader:
    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.load_numpy_files()

    def load_numpy_files(self):
        files = os.listdir(self.dir_path)
        npy_files = [file for file in files if file.endswith(".npy")]

        for npy_file in npy_files:
            attr_name = os.path.splitext(npy_file)[0]
            attr_value = np.load(os.path.join(self.dir_path, npy_file))
            setattr(self, attr_name, attr_value)

    def __repr__(self) -> str:
        return f"NumpyLoader({self.dir_path})"

    def __str__(self) -> str:
        return f"NumpyLoader({self.dir_path})"


def get_files(dataroot, ext="npy"):
    """
    Get file paths, labels, and level dictionary for a given dataroot directory.
    This is very useful for loading lots of files that are sorted in folders,
    which label the included files. The returned labels are integer values and
    the level dict shows which integer corresponds to which parent directory name.

    Parameters
    ----------
    dataroot : str
        The root directory where files are located.
    ext : str, optional
        The file extension to search for. Default is "npy".

    Returns
    -------
    tuple
        A tuple containing:
        - files : list
            A list of file paths.
        - labels : list
            A list of labels corresponding to the parent directory names of the files.
        - level_dict : dict
            A dictionary mapping unique parent directory names to integer labels.
    """

    # Get file paths
    files = list(pathlib.Path(dataroot).rglob(ext))

    # Extract parent directory names as labels
    parents = [file.parent.name for file in files]

    # Create a dictionary mapping parent directory names to integer labels
    str_levels = np.unique(parents)
    int_levels = np.arange(len(str_levels))
    level_dict = dict(zip(str_levels, int_levels))

    # Convert parent directory names to integer labels
    labels = [level_dict[parent] for parent in parents]

    # Convert posix paths to strings
    files = [str(file) for file in files]

    return files, labels, level_dict


def load_data(path: pathlib.Path, ext="npy"):
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")

    if ext in ("nix", ".nix"):
        files = [file for file in path.glob("*") if file.suffix in ext]
        if len(files) > 1:
            raise ValueError(
                "Multiple nix files found in directory! Aborting ..."
            )
        return NixDataset(files[0])

    if ext in ("npy", ".npy"):
        return NumpyDataset(path)


class DataSubset:
    def __init__(self, data, start, stop):
        self.samplerate = data.samplerate
        self.raw = data.raw[start:stop, :]
        start_t = start / self.samplerate
        stop_t = stop / self.samplerate
        self.n_electrodes = data.n_electrodes
        tracks = []
        # powers = []
        indices = []
        idents = []
        for track_id in np.unique(
            data.track_idents[~np.isnan(data.track_idents)]
        ):
            track = data.track_freqs[data.track_idents == track_id]
            # power = data.track_powers[data.track_idents == track_id, :]
            time = data.track_times[
                data.track_indices[data.track_idents == track_id]
            ]
            index = data.track_indices[data.track_idents == track_id]

            track = track[(time >= start_t) & (time <= stop_t)]
            # power = power[(time >= start_t) & (time <= stop_t), :]
            index = index[(time >= start_t) & (time <= stop_t)]
            ident = np.repeat(track_id, len(track))

            tracks.append(track)
            # powers.append(power)
            indices.append(index)
            idents.append(ident)

        # convert to numpy arrays
        tracks = np.concatenate(tracks)
        # powers = np.concatenate(powers)
        indices = np.concatenate(indices)
        idents = np.concatenate(idents)
        time = data.track_times[
            (data.track_times >= start_t) & (data.track_times <= stop_t)
        ]
        if len(indices) == 0:
            self.hasdata = False
        else:
            self.hasdata = True
            indices -= indices[0]

        self.track_freqs = tracks
        # self.track_powers = powers
        self.track_idents = idents
        self.track_indices = indices
        self.track_times = time


class NumpyDataset:
    def __init__(self, datapath: pathlib.Path) -> None:
        self.path = datapath

        # load raw file for simulated and real data
        file = os.path.join(datapath / "traces-grid1.raw")
        if os.path.exists(file):
            self.raw = DataLoader(file, 60.0, 0, channel=-1)
            self.samplerate = self.raw.samplerate
            self.n_electrodes = self.raw.shape[1]
        else:
            self.raw = np.load(datapath / "raw.npy", allow_pickle=True)
            self.samplerate = 20000.0
            if len(np.shape(self.raw)) > 1:
                self.n_electrodes = self.raw.shape[1]
            else:
                self.n_electrodes = 1
                self.raw = self.raw[:, np.newaxis]

        self.track_times = np.load(datapath / "times.npy", allow_pickle=True)
        self.track_freqs = np.load(datapath / "fund_v.npy", allow_pickle=True)
        self.track_indices = np.load(datapath / "idx_v.npy", allow_pickle=True)
        self.track_idents = np.load(datapath / "ident_v.npy", allow_pickle=True)
        # self.track_powers = np.load(datapath / "sign_v.npy", allow_pickle=True)

        # if len(self.track_powers) != len(self.track_freqs):
        #     raise ValueError(
        #         "Number of tracks and number of powers do not match! Fix dataset!"
        #     )

    def __repr__(self) -> str:
        return f"NumpyDataset({self.file})"

    def __str__(self) -> str:
        return f"NumpyDataset({self.file})"


class NixDataset:
    def __init__(self, path: pathlib.Path):
        self.path = path
        nixfile = nio.File.open(str(path), nio.FileMode.ReadOnly)

        if len(nixfile.blocks) > 1:
            print("File contains more than one block. Using first block only.")

        file = os.path.join(path / "traces-grid1.raw")
        self.raw = DataLoader(file, 60.0, 0, channel=-1)

        block = nixfile.blocks[0]
        self.track_freqs = np.asarray(block.data_arrays["track_freqs"][:])
        self.track_times = np.asarray(block.data_arrays["track_times"][:])
        self.track_idents = np.asarray(block.data_arrays["track_idents"][:])
        self.track_indices = np.asarray(block.data_arrays["track_indices"][:])
        # self.spec = block.data_arrays["spec"]
        # self.spec_freqs = block.data_arrays["spec_freq"][:]
        # self.spec_times = np.asarray(block.data_arrays["spec_time"][:])

    def __repr__(self) -> str:
        return f"NixDataset({self.path})"

    def __str__(self) -> str:
        return f"NixDataset({self.path})"

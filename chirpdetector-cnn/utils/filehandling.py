import os
import pathlib
from typing import Union

import numpy as np
import yaml
from IPython import embed
from rich import print
from thunderfish.dataloader import DataLoader


def todict(obj, classkey=None):
    """Recursively convert an object into a dictionary.

    Parameters
    ----------
    obj : _object_
        Some object to convert into a dictionary.
    classkey : str, optional
        The key to that should be converted. If None,
        converts everything in the object. By default None

    Returns
    -------
    dict
        The converted dictionary.
    """
    if isinstance(obj, dict):
        data = {}
        for k, v in obj.items():
            data[k] = todict(v, classkey)
        return data
    elif hasattr(obj, "_ast"):
        return todict(obj._ast())
    elif hasattr(obj, "__iter__") and not isinstance(obj, str):
        return [todict(v, classkey) for v in obj]
    elif hasattr(obj, "__dict__"):
        data = dict(
            [
                (key, todict(value, classkey))
                for key, value in obj.__dict__.items()
                if not callable(value) and not key.startswith("_")
            ]
        )
        if classkey is not None and hasattr(obj, "__class__"):
            data[classkey] = obj.__class__.__name__
        return data
    else:
        return obj


class Config:
    """
    Class to recursively load a YAML file and access its contents using
    dot notation.

    Parameters
    ----------
    config_file : str
        The path to the YAML file to load.

    Attributes
    ----------
    <key> : Any
        The value associated with the specified key in the loaded YAML file. If the value is
        a dictionary, it will be recursively converted to another `Config` object and accessible
        as a subclass attribute.
    """

    def __init__(self, config_file: Union[str, pathlib.Path, dict]) -> None:
        """
        Load the YAML file and convert its keys to class attributes.
        """

        if isinstance(config_file, dict):
            config_dict = config_file
        else:
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:
        """
        Return a string representation of the `Config` object.
        """
        return f"Config({vars(self)})"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the `Config` object.
        """
        return str(vars(self))

    def pprint(self) -> None:
        """
        Pretty print the `Config` object.
        """
        print(todict(self))


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

    if ext in ("npy", ".npy"):
        return NumpyDataset(path)


class DataSubset:
    def __init__(self, data, start, stop):
        self.samplerate = data.samplerate
        self.path = data.path
        self.raw = data.raw[start:stop, :]
        start_t = start / self.samplerate
        stop_t = stop / self.samplerate
        self.n_electrodes = data.n_electrodes
        tracks = []
        indices = []
        idents = []
        for track_id in np.unique(
            data.track_idents[~np.isnan(data.track_idents)]
        ):
            track = data.track_freqs[data.track_idents == track_id]
            time = data.track_times[
                data.track_indices[data.track_idents == track_id]
            ]
            index = data.track_indices[data.track_idents == track_id]

            track = track[(time >= start_t) & (time <= stop_t)]
            index = index[(time >= start_t) & (time <= stop_t)]
            ident = np.repeat(track_id, len(track))

            tracks.append(track)
            indices.append(index)
            idents.append(ident)

        # convert to numpy arrays
        tracks = np.concatenate(tracks)
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

    def __repr__(self) -> str:
        return f"NumpyDataset({self.file})"

    def __str__(self) -> str:
        return f"NumpyDataset({self.file})"


class WaveTrackerDataset:
    def __init__(self, datapath: pathlib.Path) -> None:
        if not datapath.is_dir():
            raise NotADirectoryError(f"{datapath} is not a directory")

        self.path = datapath
        self.date = datapath.name

        if pathlib.Path(datapath / "raw.npy").exists():
            self.raw = np.load(datapath / "raw.npy", allow_pickle=True)
            self.samplerate = 20000.0
        elif pathlib.Path(datapath / "traces-grid1.raw").exists():
            self.raw = DataLoader(
                datapath / "traces-grid1.raw", 60.0, 0, channel=-1
            )
            self.samplerate = self.raw.samplerate
        else:
            raise FileNotFoundError(
                f"Could not find raw data file in {datapath}"
            )

        self.n_electrodes = self.raw.shape[1]

        self.track_times = np.load(datapath / "times.npy", allow_pickle=True)
        self.track_freqs = np.load(datapath / "fund_v.npy", allow_pickle=True)
        self.track_indices = np.load(datapath / "idx_v.npy", allow_pickle=True)
        self.track_idents = np.load(datapath / "ident_v.npy", allow_pickle=True)
        self.track_powers = np.load(datapath / "sign_v.npy", allow_pickle=True)
        self.ids = np.unique(self.track_idents[~np.isnan(self.track_idents)])

    def save(self, path: pathlib.Path | None = None):
        if path is None:
            path = self.path
        if not path.is_dir():
            try:
                path.mkdir(parents=True)
                print(f"Created directory {path}")
            except FileExistsError:
                print(f"{path} already exists, aborting save")

        np.save(path / "raw.npy", self.raw)
        np.save(path / "times.npy", self.track_times)
        np.save(path / "fund_v.npy", self.track_freqs)
        np.save(path / "idx_v.npy", self.track_indices)
        np.save(path / "ident_v.npy", self.track_idents)
        np.save(path / "sign_v.npy", self.track_powers)

    def __repr__(self) -> str:
        return f"WaveTrackerDataset({self.file})"

    def __str__(self) -> str:
        return f"WaveTrackerDataset({self.file})"


class WaveTrackerDataSubset:
    def __init__(
        self,
        dataset: WaveTrackerDataset,
        start: Union[int, float],
        stop: Union[int, float],
        on: str = "index",
    ) -> None:
        assert on in ("index", "time"), "on must be either 'index' or 'time'"

        self.samplerate = dataset.samplerate
        self.path = dataset.path
        self.date = dataset.date
        self.n_electrodes = dataset.n_electrodes

        if on == "index":
            assert (
                start < stop
            ), "start must be smaller than stop when on='index'"
            assert start >= 0, "start must be larger than 0 when on='index'"
            assert (
                stop <= dataset.raw.shape[0]
            ), "stop must be smaller than the number of samples in the dataset"
            assert isinstance(
                start, int
            ), "start must be an integer when on='index'"
            assert isinstance(
                stop, int
            ), "stop must be an integer when on='index'"
            self.start_idx = start
            self.stop_idx = stop
            self.start_t = self.start_idx / self.samplerate
            self.stop_t = self.fstop_idx / self.samplerate
        if on == "time":
            assert (
                start < stop
            ), "start must be smaller than stop when on='time'"
            assert start >= 0, "start must be larger than 0"
            assert (
                stop <= dataset.raw.shape[0] / self.samplerate
            ), "stop must be smaller than the end time of the dataset"
            assert isinstance(
                start, (int, float)
            ), "start must be an integer or float."
            assert isinstance(
                stop, (int, float)
            ), "stop must be an integer or float."

            self.start_t = start
            self.stop_t = stop
            self.start_idx = int(np.round(self.start_t * self.samplerate))
            self.stop_idx = int(np.round(self.stop_t * self.samplerate))

        self.raw = dataset.raw[self.start_idx : self.stop_idx, :]

        self.track_freqs = []
        self.track_indices = []
        self.track_idents = []
        self.track_powers = []
        track_start_idx = np.arange(len(dataset.track_times))[
            dataset.track_times >= self.start_t
        ][0]
        for track_id in dataset.ids:
            i = dataset.track_indices[dataset.track_idents == track_id]
            f = dataset.track_freqs[dataset.track_idents == track_id]
            p = dataset.track_powers[dataset.track_idents == track_id]
            t = dataset.track_times[i]

            f = f[(t >= self.start_t) & (t <= self.stop_t)]
            p = p[(t >= self.start_t) & (t <= self.stop_t)]
            i = i[(t >= self.start_t) & (t <= self.stop_t)] - track_start_idx
            t = t[(t >= self.start_t) & (t <= self.stop_t)] - self.start_t
            ids = np.ones_like(t) * track_id

            self.track_freqs.append(f)
            self.track_indices.append(i)
            self.track_idents.append(ids)
            self.track_powers.append(p)

        self.track_times = (
            dataset.track_times[
                (dataset.track_times >= self.start_t)
                & (dataset.track_times <= self.stop_t)
            ]
            - self.start_t
        )

        self.track_freqs = np.concatenate(self.track_freqs)
        self.track_indices = np.concatenate(self.track_indices)
        self.track_idents = np.concatenate(self.track_idents)
        self.track_powers = np.concatenate(self.track_powers)
        self.ids = np.unique(self.track_idents[~np.isnan(self.track_idents)])

    def save(self, path: pathlib.Path | None = None):
        if path is None:
            path = self.path
        if not path.is_dir():
            try:
                path.mkdir()
                print('Created directory "{path}"')
            except FileExistsError:
                print(f"{path} already exists, skipping save.")

        np.save(path / "raw.npy", self.raw)
        np.save(path / "times.npy", self.track_times)
        np.save(path / "fund_v.npy", self.track_freqs)
        np.save(path / "idx_v.npy", self.track_indices)
        np.save(path / "ident_v.npy", self.track_idents)
        np.save(path / "sign_v.npy", self.track_powers)

    def __repr__(self) -> str:
        return f"WaveTrackerDataSubset({self.path})"

    def __str__(self) -> str:
        return f"WaveTrackerDataSubset({self.path})"


class ChirpDataset(WaveTrackerDataset):
    def __init__(self, datapath: pathlib.Path) -> None:
        super().__init__(datapath)

        if pathlib.Path(datapath / "chirp_times_cnn.npy").exists() == False:
            raise FileNotFoundError(
                f"Could not find chirp data file in {datapath}"
            )

        self.chirp_times = np.load(datapath / "chirp_times_cnn.npy")
        self.chirp_ids = np.load(datapath / "chirp_ids_cnn.npy")

    def save(self, path: pathlib.Path | None):
        if path is None:
            path = self.path
        if not path.is_dir():
            try:
                path.mkdir(parents=True)
                print(f"Created directory {path}")
            except FileExistsError:
                print(f"{path} already exists, aborting save")

        super.save(path)
        np.save(path / "chirp_times_cnn.npy", self.chirp_times)
        np.save(path / "chirp_ids_cnn.npy", self.chirp_ids)

    def __repr__(self) -> str:
        return f"ChirpDataset({self.file})"

    def __str__(self) -> str:
        return f"ChirpDataset({self.file})"


class ChirpDataSubset(WaveTrackerDataSubset):
    def __init__(
        self,
        dataset: ChirpDataset,
        start: int | float,
        stop: int | float,
        on: str = "index",
    ) -> None:
        super().__init__(dataset, start, stop, on)

        self.chirp_ids = dataset.chirp_ids[
            (dataset.chirp_times >= self.start_t)
            & (dataset.chirp_times <= self.stop_t)
        ]
        self.chirp_times = dataset.chirp_times[
            (dataset.chirp_times >= self.start_t)
            & (dataset.chirp_times <= self.stop_t)
        ]
        self.chirp_times -= self.start_t

    def save(self, path: pathlib.Path | None = None):
        if path is None:
            path = self.path
        if not path.is_dir():
            try:
                path.mkdir(parents=True)
                print(f"Created directory {path}")
            except FileExistsError:
                print(f"{path} already exists, aborting save")

        super().save(path)
        np.save(path / "chirp_times_cnn.npy", self.chirp_times)
        np.save(path / "chirp_ids_cnn.npy", self.chirp_ids)

    def __repr__(self) -> str:
        return f"ChirpDataSubset({self.path})"

    def __str__(self) -> str:
        return f"ChirpDataSubset({self.path})"

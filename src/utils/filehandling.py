import os
import pathlib

import nixio as nio
import numpy as np
import yaml
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


# class NumpyLoader:
#     def __init__(self, dir_path):
#         self.dir_path = dir_path
#         self.load_numpy_files()

#     def load_numpy_files(self):
#         files = os.listdir(self.dir_path)
#         npy_files = [file for file in files if file.endswith(".npy")]

#         for npy_file in npy_files:
#             attr_name = os.path.splitext(npy_file)[0]
#             attr_value = np.load(os.path.join(self.dir_path, npy_file))
#             setattr(self, attr_name, attr_value)

#     def info(self):
#         pprint(vars(self))

#     def __repr__(self) -> str:
#         return f"NumpyLoader({self.dir_path})"

#     def __str__(self) -> str:
#         return f"NumpyLoader({self.dir_path})"


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


class NumpyDataset:
    def __init__(self, datapath: pathlib.Path) -> None:
        self.path = datapath
        file = os.path.join(datapath / "traces-grid1.raw")
        self.raw = DataLoader(file, 60.0, 0, channel=-1)
        self.raw_rate = self.raw.samplerate
        self.track_times = np.load(datapath / "times.npy", allow_pickle=True)
        self.track_freqs = np.load(datapath / "fund_v.npy", allow_pickle=True)
        self.track_indices = np.load(datapath / "idx_v.npy", allow_pickle=True)
        self.track_idents = np.load(datapath / "ident_v.npy", allow_pickle=True)

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

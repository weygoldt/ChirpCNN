import os
import pathlib
from pprint import pprint

import nixio as nio
import numpy as np
import yaml

# from logger import make_logger
from thunderfish.dataloader import DataLoader

# logger = make_logger(__name__)


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

    def info(self):
        pprint(vars(self))

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


class LoadData:
    """
    Load data from raw file and wavetracker files

    Attributes
    ----------
    data : DataLoader object containing raw data
    samplerate : sampling rate of raw data
    time : array of time for tracked fundamental frequency
    freq : array of fundamental frequency
    idx : array of indices to access time array
    ident : array of identifiers for each tracked fundamental frequency
    ids : array of unique identifiers exluding NaNs
    """

    def __init__(self, datapath: pathlib.Path) -> None:
        # load raw data
        self.datapath = datapath
        self.file = os.path.join(datapath / "/traces-grid1.raw")
        self.raw = DataLoader(self.file, 60.0, 0, channel=-1)
        self.raw_rate = self.raw.samplerate

        # load wavetracker files
        self.time = np.load(datapath / "/times.npy", allow_pickle=True)
        self.freq = np.load(datapath / "/fund_v.npy", allow_pickle=True)
        self.powers = np.load(datapath / "/sign_v.npy", allow_pickle=True)
        self.idx = np.load(datapath / "/idx_v.npy", allow_pickle=True)
        self.ident = np.load(datapath / "/ident_v.npy", allow_pickle=True)
        self.ids = np.unique(self.ident[~np.isnan(self.ident)])

    def __repr__(self) -> str:
        return f"LoadData({self.file})"

    def __str__(self) -> str:
        return f"LoadData({self.file})"

import yaml
import numpy as np
from pathlib import Path


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
    files = list(Path(dataroot).rglob(ext))

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


class load_conf():
    def __init__(self, path):
        self.path = path


#!/usr/bin/env python3

"""
This module contains the dataset types that the detector can load
to detect chirps in grid recordings. NIX files are the default.
"""

import pathlib
from pprint import pprint

import nixio as nio
import numpy as np
from IPython import embed


class NixDataset:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.file = nio.File.open(str(path), nio.FileMode.ReadOnly)

        if len(self.file.blocks) > 1:
            print("File contains more than one block. Using first block only.")

        block = self.file.blocks[0]
        self.spec = block.data_arrays["spec"]
        self.spec_freqs = block.data_arrays["spec_freq"][:]
        self.spec_times = np.asarray(block.data_arrays["spec_time"][:])
        self.track_freqs = np.asarray(block.data_arrays["track_freqs"][:])
        self.track_times = np.asarray(block.data_arrays["track_times"][:])
        self.track_idents = np.asarray(block.data_arrays["track_idents"][:])
        self.track_indices = np.asarray(block.data_arrays["track_indices"][:])

    def __repr__(self) -> str:
        return f"NixDataset({self.path})"

    def __str__(self) -> str:
        return f"NixDataset({self.path})"


class NumpyDataset:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.spec = np.load(path / "fill_spec.npy", allow_pickle=True)
        self.spec_freqs = np.load(path / "fill_freqs.npy")
        self.spec_times = np.load(path / "fill_times.npy")
        self.track_freqs = np.load(path / "fund_v.npy")
        self.track_times = np.load(path / "times.npy")
        self.track_idents = np.load(path / "ident_v.npy")
        self.track_indices = np.load(path / "idx_v.npy")

    def __repr__(self) -> str:
        return f"NumpyDataset({self.path})"

    def __str__(self) -> str:
        return f"NumpyDataset({self.path})"


def get_files(path: pathlib.Path, ext: str = (".npy", ".nix")):
    files = [file for file in path.glob("*") if file.suffix in ext]
    extensions = set(file.suffix for file in files)
    return files, extensions


def load_data(path: pathlib.Path):
    if not path.is_dir():
        raise NotADirectoryError(f"{path} is not a directory")

    files, extensions = get_files(path)

    if ".nix" in extensions:
        nixfiles = [file for file in files if file.suffix == ".nix"]
        if len(nixfiles) > 1:
            raise ValueError(
                "Multiple nix files found in directory! Aborting ..."
            )
        return NixDataset(nixfiles[0])

    elif ".npy" in extensions:
        return NumpyDataset(path)


def main():
    path = pathlib.Path("../../real_data")
    data = load_data(path)
    pprint(dir(data))
    embed()
    exit()


if __name__ == "__main__":
    main()

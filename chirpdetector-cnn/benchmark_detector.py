#!/usr/bin/env python3

"""
Detect chirps on a benchmark dataset and compare the detected chrip times 
for each id with the ground truth.
"""

import argparse
import pathlib

import numpy as np
from assign_chirps import assign_chirps
from detect_chirps import Detector
from rich import pretty, print
from rich.progress import track
from utils.filehandling import (
    ChirpDataset,
    ConfLoader,
    NumpyDataset,
    NumpyLoader,
)
from utils.logger import make_logger
from utils.plotstyle import PlotStyle

conf = ConfLoader("config.yml")
logger = make_logger(__name__)
ps = PlotStyle()


def benchmark(path):
    modelpath = conf.save_dir
    gt = NumpyLoader(path)
    d = NumpyDataset(path)
    chirp_times = np.load(path / "chirp_times_cnn.npy")
    chirp_idents = np.load(path / "chirp_ids_cnn.npy")
    cd = ChirpDataset(path)

    tolerance = 0.04

    precs = []
    recs = []
    accs = []
    errs = []
    for fish_id in track(
        np.unique(gt.chirp_ids), description="Benchmarking ..."
    ):
        real_chirps = gt.chirp_times[gt.chirp_ids == fish_id]
        detected_chirps = chirp_times[chirp_idents == fish_id]

        # check for false negatives
        fn_counter = 0
        for rc in real_chirps:
            if not any(np.abs(rc - detected_chirps) < tolerance):
                fn_counter += 1

        # check for false positives
        fp_counter = 0
        for dc in detected_chirps:
            if not any(np.abs(dc - real_chirps) < tolerance):
                fp_counter += 1

        # compute precision, recall, accuracy, error
        precs.append(len(real_chirps) / (len(real_chirps) + fp_counter))
        recs.append(len(detected_chirps) / (len(detected_chirps) + fn_counter))

    print("")
    print("------------------------------------")
    print("Results:")
    print(
        f"Found {np.round(len(chirp_times)/len(gt.chirp_times) * 100,2)} % of total chirps"
    )
    print(
        f"Proportion of detections that are correct (Precision): {np.round(np.mean(precs)*100, 4)} %"
    )
    print(
        f"Proportion of existing chirps that are detected (Recall): {np.round(np.mean(recs)*100, 4)} %"
    )
    print(
        f"F1 score: {np.round(2*np.mean(precs)*np.mean(recs)/(np.mean(precs)+np.mean(recs))*100, 4)}"
    )
    print("------------------------------------")
    return precs, recs


def interface():
    parser = argparse.ArgumentParser(description="Benchmark chirp detector")
    parser.add_argument(
        "--path", "-p", type=pathlib.Path, help="Path to dataset"
    )
    args = parser.parse_args()
    return args


def main():
    args = interface()
    precs, recs = benchmark(args.path)


if __name__ == "__main__":
    main()

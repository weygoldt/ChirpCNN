#!/usr/bin/env python3

"""
Detect chirps on a benchmark dataset and compare the detected chrip times 
for each id with the ground truth.
"""

from pathlib import Path

import numpy as np
from detect_chirps import Detector
from rich import pretty
from rich.progress import track
from utils.filehandling import ConfLoader, NumpyDataset, NumpyLoader
from utils.logger import make_logger
from utils.plotstyle import PlotStyle

conf = ConfLoader("config.yml")
logger = make_logger(__name__)
ps = PlotStyle()
pretty.install()


def benchmark():
    path = Path(conf.testing_data_path)
    modelpath = conf.save_dir

    gt = NumpyLoader(path)
    d = NumpyDataset(path)
    det = Detector(modelpath, d)
    chirp_times, chirp_idents = det.detect()

    tolerance = 0.04

    precs = []
    recs = []
    accs = []
    errs = []
    for fish_id in track(
        np.unique(gt.correct_chirp_time_ids), description="Benchmarking ..."
    ):
        real_chirps = gt.correct_chirp_times[
            gt.correct_chirp_time_ids == fish_id
        ]
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
        f"Found {np.round(len(chirp_times)/len(gt.correct_chirp_times) * 100,2)} % of total chirps"
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


def main():
    precs, recs = benchmark()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""
Count chirps in recordings on the server. Just to know how much we have now.
"""

import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from detect_chirps import main as chirpdetector
from rich import print
from utils.logger import make_logger

logger = make_logger(__name__)


def main():
    datapath = pathlib.Path("../data/")
    meta = pd.read_csv(datapath / "order_meta.csv")

    # only use the ones with 3 or more recordings
    meta = meta[meta.group > 2]

    # only use the ones where id1 or id2 is not nan
    meta = meta[~meta.rec_id1.isna() | ~meta.rec_id2.isna()]

    # extract the paths to the recordings
    # and strip the FUCKING QUOTATION MARKS
    recs = [datapath / p.strip("“”") for p in meta.recording.values]

    counter = 0
    for i, rec in enumerate(recs):
        try:
            chirps = np.load(rec / "chirp_times_cnn.npy")
            counter += len(chirps)
            print(f"Processing recording {i+1}/{len(recs)+1}")
        except:
            print("Nothing here yet!")

    print(f"We have {counter} chirps now!")


if __name__ == "__main__":
    main()

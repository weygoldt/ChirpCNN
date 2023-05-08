#!/usr/bin/env python3

"""
Parse a dataset on the server, generate hybrid recordings and extract
training data from them.
"""

import pathlib

import pandas as pd
from training_data_from_dataset import parse_dataset
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

    for i, rec in enumerate(recs):
        logger.info(f"Processing recording {i+1}/{len(recs)+1}")
        parse_dataset(rec)


if __name__ == "__main__":
    main()

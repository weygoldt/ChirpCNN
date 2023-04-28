import os
import pathlib
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, datapath: pathlib.Path) -> None:
        self.track_times = np.load(datapath / "times.npy", allow_pickle=True)
        self.track_freqs = np.load(datapath / "fund_v.npy", allow_pickle=True)
        self.track_idents = np.load(datapath / "ident_v.npy", allow_pickle=True)
        self.track_indices = np.load(datapath / "idx_v.npy", allow_pickle=True)


datapath = pathlib.Path("../data/")
meta = pd.read_csv(datapath / "order_meta.csv")

# only use the ones with 3 or more recordings
meta = meta[meta.group > 2]

# only use the ones where id1 or id2 is not nan
meta = meta[~meta.rec_id1.isna() | ~meta.rec_id2.isna()]

# extract the paths to the recordings
# and strip the FUCKING QUOTATION MARKS
recs = [datapath / p.strip("“”") for p in meta.recording.values]

for rec in recs:
    data = Dataset(rec)
    print(rec)

    fig, axs = plt.subplots(2, 1)
    track_ids = np.unique(data.track_idents[~np.isnan(data.track_idents)])

    for track_id in track_ids:
        f = data.track_freqs[data.track_idents == track_id]
        t = data.track_times[data.track_indices[data.track_idents == track_id]]
        axs[0].plot(t, f)
        axs[1].plot(t)
    plt.show()

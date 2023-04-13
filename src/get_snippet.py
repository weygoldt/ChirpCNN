import pathlib

import matplotlib.pyplot as plt
import numpy as np
from thunderfish.powerspectrum import decibel, spectrogram

from utils.filehandling import LoadData

datapath = "/home/weygoldt/projects/chirpdetector/data/2022-06-02-10_00/"

data = LoadData(str(datapath))

# good chirp times for data: 2022-06-02-10_00
window_start_index = (3 * 60 * 60 + 6 * 60 + 20) * data.raw_rate
window_duration_index = 120 * data.raw_rate
window_stop_index = window_start_index + window_duration_index
window_start_seconds = window_start_index / data.raw_rate
window_duration_seconds = window_duration_index / data.raw_rate
window_stop_seconds = window_stop_index / data.raw_rate

raw = data.raw[
    window_start_index : window_start_index + window_duration_index, :
]

# get trace arrays
tracks = []
ident = []
idx = []

fulltime = data.time[
    (data.time >= window_start_seconds) & (data.time <= window_stop_seconds)
]
fullindex = np.arange(len(fulltime))

for track_id in np.unique(data.ident):
    # get data for fish
    track = data.freq[data.ident == track_id]
    time = data.time[data.idx[data.ident == track_id]]
    indices = data.idx[data.ident == track_id]

    # prune to match window
    new_track = track[
        (time >= window_start_seconds) & (time <= window_stop_seconds)
    ]
    indices = indices[
        (time >= window_start_seconds) & (time <= window_stop_seconds)
    ]
    fish_ids = np.full(len(new_track), track_id)

    tracks.append(new_track)
    idx.append(indices)
    ident.append(fish_ids)

tracks = np.concatenate(tracks)
idx = np.concatenate(idx)
ident = np.concatenate(ident)
time = data.time - window_start_seconds
time_shit = time[time < 0]
idx -= len(time_shit)
time = time[time >= 0]

# compute spectra
for channel in range(raw.shape[1]):
    spec, freqs, times = spectrogram(
        data=raw[:, channel],
        ratetime=data.raw_rate,
        overlap_frac=0.99,
        freq_resolution=5,
    )
    if channel == 0:
        spec_sum = spec
    else:
        spec_sum += spec

spec_sum = decibel(spec_sum)

plt.imshow(
    spec_sum,
    aspect="auto",
    origin="lower",
    extent=[np.min(times), np.max(times), np.min(freqs), np.max(freqs)],
    interpolation="none",
)
for track_id in np.unique(ident):
    t = time[idx[ident == track_id]]
    t = t - t[0]
    plt.plot(
        t,
        tracks[ident == track_id],
        "k-",
        linewidth=2,
    )
plt.ylim(500, 1200)
plt.show()

# save the data
np.save("../real_data/fill_spec.npy", spec_sum)
np.save("../real_data/fill_times.npy", times)
np.save("../real_data/fill_freqs.npy", freqs)
np.save("../real_data/fund_v.npy", tracks)
np.save("../real_data/ident_v.npy", ident)
np.save("../real_data/idx_v.npy", idx)
np.save("../real_data/times.npy", time)

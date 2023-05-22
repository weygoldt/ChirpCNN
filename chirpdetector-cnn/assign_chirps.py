#!/usr/bin/env python3

"""
Assing and correct detected chirps by checking for an amplitude trough in the 
temporal vicinity of a detected chirp.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from rich import print
from rich.progress import track
from scipy.signal import find_peaks
from utils.filehandling import ChirpDataset, Config
from utils.filters import bandpass_filter, envelope
from utils.logger import make_logger
from utils.plotstyle import PlotStyle

logger = make_logger(__name__)
ps = PlotStyle()
conf = Config("config.yml")


def assign_chirps(data):
    logger.info("Assigning chirps by amplitude trough ...")
    sorted_chirps = []
    sorted_chirp_ids = []
    for track_id in track(data.ids, description=f"Assigning chirps"):
        chirps = data.chirp_times[data.chirp_ids == track_id]
        time = data.track_times[
            data.track_indices[data.track_idents == track_id]
        ]
        freq = data.track_freqs[data.track_idents == track_id]
        power = data.track_powers[data.track_idents == track_id, :]

        for chirp in chirps:
            closest = np.argmin(np.abs(time - chirp))
            best_electrode = np.argmax(power[closest, :])
            best_freq = freq[closest]

            start = int(
                np.round((chirp - conf.assign.time_range) * data.samplerate)
            )
            stop = int(
                np.round((chirp + conf.assign.time_range) * data.samplerate)
            )
            lower_f = best_freq - conf.assign.freq_range
            upper_f = best_freq + conf.assign.freq_range

            raw = data.raw[start:stop, best_electrode]
            if len(raw) == 0:
                continue

            raw_filtered = bandpass_filter(
                signal=raw,
                samplerate=data.samplerate,
                lowf=lower_f,
                highf=upper_f,
            )

            env = envelope(
                signal=raw_filtered,
                samplerate=data.samplerate,
                cutoff_frequency=conf.assign.env_cutoff,
            )

            # cut off the first and last 10% of the envelope to
            # remove edge effects
            new_start = int(np.round(len(env) * 0.1))
            new_stop = int(np.round(len(env) * 0.9))
            t = np.arange(start, stop)[new_start:new_stop] / data.samplerate
            env = -env[new_start:new_stop]

            peaks = find_peaks(env)[0]

            # check if there is a peak in the vicinity of the chirp
            if np.any(np.abs(t[peaks] - chirp) < conf.assign.time_tolerance):
                sorted_chirps.append(chirp)
                sorted_chirp_ids.append(track_id)

    sorted_chirps = np.array(sorted_chirps)
    sorted_chirp_ids = np.array(sorted_chirp_ids)

    np.save(data.path / "chirp_times_cnn.npy", sorted_chirps)
    np.save(data.path / "chirp_ids_cnn.npy", sorted_chirp_ids)


def interface():
    parser = argparse.ArgumentParser(
        description="Assign and correct detected chirps."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=pathlib.Path,
        help="Path to the directory containing the data.",
    )
    args = parser.parse_args()

    return args


def main(path: pathlib.Path):
    data = ChirpDataset(path)
    assign_chirps(data)


if __name__ == "__main__":
    args = interface()
    main(args.path)

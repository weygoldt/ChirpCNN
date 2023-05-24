#!/usr/bin/env python3

"""
Allows to correct the detected chirp times by hand using a minimal matplotlib
interface.
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from rich import print
from rich.progress import track
from utils.datahandling import find_on_time
from utils.filehandling import ChirpDataset, Config
from utils.plotstyle import PlotStyle
from utils.spectrogram import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    specshow,
    spectrogram,
)

conf = Config("config.yml")
ps = PlotStyle()


def interactive_plot(plot_data, dataset):
    fig, ax = plt.subplots()
    specshow(
        plot_data["spec"],
        plot_data["spec_times"],
        plot_data["spec_freqs"],
        ax=ax,
        aspect="auto",
        origin="lower",
    )
    detected_points = []
    for track_id in np.unique(dataset.track_idents):
        t = dataset.track_times[dataset.track_indices[dataset.track_idents == track_id]]
        f = dataset.track_freqs[dataset.track_idents == track_id]

        f = f[(t >= plot_data["spec_times"][0]) & (t <= plot_data["spec_times"][-1])]
        t = t[(t >= plot_data["spec_times"][0]) & (t <= plot_data["spec_times"][-1])]

        ax.plot(t, f, color="black", linewidth=1)

        chirp_x = plot_data["chirps"][plot_data["chirps"][:, 1] == track_id, 0]
        y_indices = [find_on_time(t, x) for x in chirp_x]
        chirp_y = f[y_indices]
        detected_points.append(np.asarray([chirp_x, chirp_y]).T)

    detected_points = np.concatenate(detected_points, axis=0).tolist()
    scatter = ax.scatter(
        np.array(detected_points)[:, 0],
        np.array(detected_points)[:, 1],
        marker=".",
        edgecolors="black",
        facecolors="white",
    )

    def on_click(event):
        if event.button == 1:  # Left mouse button to add a point
            new_point = [event.xdata, event.ydata]
            detected_points.append(new_point)
            scatter.set_offsets(detected_points)
        elif event.button == 3:  # Right mouse button to remove a point
            if len(detected_points) > 0:
                distances = (
                    (np.array(detected_points) - [event.xdata, event.ydata]) ** 2
                ).sum(axis=1)
                closest_index = distances.argmin()
                detected_points.pop(closest_index)
                scatter.set_offsets(detected_points)
        fig.canvas.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    plt.show()

    # assign back ids to the chirp times
    assigned_points = []
    for point in detected_points:
        point_t = point[0]
        point_f = point[1]
        distances = []
        ids = []
        for track_id in np.unique(dataset.track_idents[~np.isnan(dataset.track_idents)]):
            f = dataset.track_freqs[dataset.track_idents == track_id]
            t = dataset.track_times[
                dataset.track_indices[dataset.track_idents == track_id]
            ]

            closest_index = find_on_time(t, point_t)
            closest_freq = f[closest_index]
            distances.append(np.abs(closest_freq - point_f))
            ids.append(track_id)

        distances = np.array(distances)
        closest_id = ids[np.argmin(distances)]
        point[1] = closest_id
        assigned_points.append(point)

    assigned_points = np.asarray(assigned_points)
    assigned_points = assigned_points[assigned_points[:, 0].argsort()]

    return assigned_points


def correct_chirps(path):
    cd = ChirpDataset(path)

    nfft = freqres_to_nfft(conf.frequency_resolution, cd.samplerate)
    hop_len = overlap_to_hoplen(conf.overlap_fraction, nfft)

    # group chirps to make chunks that match the snippet size from the config
    buffersize = conf.buffersize * 3

    # make chirp groups that last as long as the buffer size
    groups = []
    group = []
    cd.chirp_ids = cd.chirp_ids[cd.chirp_times.argsort()]
    cd.chirp_times = cd.chirp_times[cd.chirp_times.argsort()]
    for chirp_time, chirp_id in zip(cd.chirp_times, cd.chirp_ids):
        if group == []:
            group.append((chirp_time, chirp_id))
        elif chirp_time - group[0][0] < buffersize:
            group.append((chirp_time, chirp_id))
        else:
            groups.append(group)
            group = []
            group.append((chirp_time, chirp_id))

    # plot each group in a interactive matplotlib window
    chirp_times = []
    chirp_ids = []
    for i, group in enumerate(groups):
        group = np.asarray(group)
        start_t = group[0][0] - conf.spectrogram_overlap
        if start_t < 0:
            start_t = 0
        stop_t = group[-1][0] + conf.spectrogram_overlap

        start_i = int(np.round(start_t * cd.samplerate))
        stop_i = int(np.round(stop_t * cd.samplerate))

        spec_times = np.arange(start_i, stop_i + 1, hop_len) / cd.samplerate
        spec_freqs = np.arange(0, nfft / 2 + 1) * cd.samplerate / nfft

        for el in range(cd.n_electrodes):
            sig = cd.raw[start_i:stop_i, el]
            chunk_spec, _, _ = spectrogram(
                sig.copy(),
                cd.samplerate,
                nfft=nfft,
                hop_length=hop_len,
            )

            # sum spectrogram over all electrodes
            # the spec is a tensor
            if el == 0:
                spec = chunk_spec
            else:
                spec += chunk_spec

        # normalize spectrogram by the number of electrodes
        # the spec is still a tensor
        spec /= cd.n_electrodes

        # convert the spectrogram to dB
        # .. still a tensor
        spec = decibel(spec)

        # cut off everything outside the upper frequency limit
        # the spec is still a tensor
        spec = spec[spec_freqs <= conf.upper_spectrum_limit, :]
        spec_freqs = spec_freqs[spec_freqs <= conf.upper_spectrum_limit]

        plot_data = {
            "spec": spec.cpu().numpy(),
            "spec_freqs": spec_freqs,
            "spec_times": spec_times,
            "chirps": group,
        }

        assigned_chirps = interactive_plot(plot_data, cd)

        chirp_times.append(assigned_chirps[:, 0])
        chirp_ids.append(assigned_chirps[:, 1])

    chirp_times = np.concatenate(chirp_times)
    chirp_ids = np.concatenate(chirp_ids)
    np.save(
        path / "chirp_times_gt.npy",
        chirp_times,
    )
    np.save(
        path / "chirp_ids_gt.npy",
        chirp_ids,
    )
    print("Saved corrected chirp times and ids.")


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", "-p", type=pathlib.Path, help="Path to the dataset.")
    args = parser.parse_args()
    return args


def main():
    args = interface()
    correct_chirps(args.path)


if __name__ == "__main__":
    main()

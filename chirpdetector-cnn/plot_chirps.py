#!/usr/bin/env python3

"""
Plot the chirps from the spectrograms that are computed and, if enabled, saved
during the detection process. Plotting is outsourced from the detection process
to allow for the visualization of the impact of post processing steps on the
detection results.
"""

import argparse
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from rich.progress import track
from utils.datahandling import get_closest_indices, interpolate
from utils.filehandling import ChirpDataset, Config
from utils.plotstyle import PlotStyle

matplotlib.use("Agg")

ps = PlotStyle()
conf = Config("config.yml")


def plot_chirps(path):
    cd = ChirpDataset(path)
    cd = interpolate(cd, 0.02)

    specpath = path / "chirpdetector_spectrograms"
    plotpath = pathlib.Path(conf.plot_path) / path.name
    plotpath.mkdir(parents=True, exist_ok=True)

    indices = []
    files = specpath.glob("*.npy")
    for file in files:
        filename = file.stem
        indices.append(int(filename.split("_")[-1]))
    indices = np.unique(indices)

    for i in track(indices, description="Plotting chirps"):
        powers = np.load(specpath / f"spec_powers_{i}.npy")
        freqs = np.load(specpath / f"spec_freqs_{i}.npy")
        times = np.load(specpath / f"spec_times_{i}.npy")

        start, stop = times[0], times[-1]

        chirp_ids = cd.chirp_ids[
            (cd.chirp_times >= start) & (cd.chirp_times <= stop)
        ]
        chirp_times = cd.chirp_times[
            (cd.chirp_times >= start) & (cd.chirp_times <= stop)
        ]

        fig, ax = plt.subplots(
            figsize=(30 * ps.cm, 15 * ps.cm), constrained_layout=True
        )

        ax.imshow(
            powers,
            origin="lower",
            aspect="auto",
            extent=[times[0], times[-1], freqs[0], freqs[-1]],
        )

        for track_id in np.unique(cd.track_idents):
            t = cd.track_times[cd.track_indices[cd.track_idents == track_id]]
            f = cd.track_freqs[cd.track_idents == track_id]

            f = f[(t >= start) & (t <= stop)]
            t = t[(t >= start) & (t <= stop)]

            ax.plot(t, f, color=ps.black, linewidth=1)

            chirpx = chirp_times[chirp_ids == track_id]
            f_index = get_closest_indices(t, chirpx)
            chirpy = f[f_index]

            ax.scatter(
                chirpx,
                chirpy,
                s=20,
                edgecolors="black",
                facecolors="white",
                zorder=10,
            )

            ax.text(
                t[0] + 0.01,
                f[0] + 10,
                f"{track_id}",
                fontsize=8,
                color="white",
                zorder=10,
            )

        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        plt.savefig(
            plotpath / f"chirp_{i}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.cla()
        plt.clf()
        plt.close("all")
        plt.close(fig)


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path", "-p", type=pathlib.Path, help="Path to the dataset to plot."
    )
    args = parser.parse_args()
    return args


def main():
    args = interface()
    plot_chirps(args.path)


if __name__ == "__main__":
    main()

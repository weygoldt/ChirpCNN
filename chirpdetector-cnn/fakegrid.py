#!/usr/bin/env python3

"""
A first try of generating a fully simulated electrode grid recording.
Chirps, rises and distance-dependent envelopes are implemented
for each simulated electrode. The dipole is still missing. Currently,
fish are simulated as a point source that moves with a random walk.
In future, this module will be combined with a chirp simulator that
includes an undershoot and a hybrid dataset approach to generate
a ground truth for chirp detection algorithms.
"""


from typing import Tuple

import numpy as np
from IPython import embed
from simulations.fish_movement import FishMovement
from simulations.fish_signal import FishSignal
from thunderfish.efield import efish_monopoles, epotential_meshgrid
from utils.plotstyle import PlotStyle

s = PlotStyle()
np.random.seed(0)


def grid(origin, shape, spacing, type="hex"):
    assert type in ["hex", "square"], "type must be 'hex' or 'square'"

    # grid parameters
    electrode_number = shape[0] * shape[1]
    electrode_index = np.arange(0, electrode_number)
    electrode_x = np.mod(electrode_index, shape[0]) * spacing
    electrode_y = np.floor(electrode_index / shape[0]) * spacing
    # shift every second row to make a hexagonal grid
    if type == "hex":
        electrode_y[1::2] += spacing / 2

    # shift the grid to the specified origin
    electrode_x += origin[0] - np.mean(electrode_x)
    electrode_y += origin[1] - np.mean(electrode_y)

    return electrode_x, electrode_y


class Recording:
    def __init__(
        self,
        fishcount: int,
        duration: int,
        grid_shape: Tuple[int, int],
        electrode_spacing: int,
    ):
        self.samplerate = 20000

        # EOD parameters
        mineodf = 500
        maxeodf = 1000
        maxchirprate = 0.1
        maxriserate = 0.1
        nmaxchirps = int(np.floor(duration * maxchirprate))
        nmaxrises = int(np.floor(duration * maxriserate))

        # dipole model parameters
        # minsize, maxsize = 6, 18

        # save the electrode positions
        n_electrodes = grid_shape[0] * grid_shape[1]
        self.ex, self.ey = grid(
            origin=[0, 0],
            shape=grid_shape,
            spacing=electrode_spacing,
            type="hex",
        )

        self.traces = []
        self.x = []
        self.y = []

        for i in range(fishcount):
            print("simulating fish ", i + 1, " of ", fishcount)

            tmax = duration  # duration in seconds
            origin = [0, 0]  # starting point
            boundaries = [(-2, 2), (-2, 2)]  # boundaries of the arena
            mov = FishMovement(
                tmax, self.samplerate, origin, boundaries
            )  # movement object

            x, y = mov.x, mov.y  # get the position data
            traj_v = np.array(
                [np.cos(mov.trajectories), np.sin(mov.trajectories)]
            ).T
            # tail_bend = 0
            # size = np.random.randint(minsize, maxsize)

            # make the dipole matrix for every time step
            # dipole_matrix = [
            #     efish_monopoles((xi, yi), (tu, tv), size, tail_bend)
            #     for xi, yi, tu, tv in zip(x, y, traj_v[:, 0], traj_v[:, 1])
            # ]
            # print("dipole matrix len: ", len(dipole_matrix))

            # save the position data
            self.x.append(x)
            self.y.append(y)

            # calculate the distance to every electrode
            distances = np.sqrt(
                (x[:, None] - self.ex[None, :]) ** 2
                + (y[:, None] - self.ey[None, :]) ** 2
            )

            # make the distance squared and invert it
            distances = distances**2
            distances = np.max(distances) - distances

            # generate random EOD parameters
            if nmaxchirps > 0:
                nchirps = np.random.randint(0, nmaxchirps)
            else:
                nchirps = 0
            if nmaxrises > 0:
                nrises = np.random.randint(0, nmaxrises)
            else:
                nrises = 0

            eodf = np.random.randint(mineodf, maxeodf)

            # make the EOD
            sig = FishSignal(self.samplerate, duration, eodf, nchirps, nrises)

            # put the signal on every electrode
            grid_signals = np.tile(sig.signal, (n_electrodes, 1)).T

            # attenuate the signal with the distance
            attenuated_signals = grid_signals * distances

            # normalize by the number of simulated fish
            attenuated_signals = attenuated_signals / fishcount

            # add the signal to the recording
            if i == 0:
                self.signal = attenuated_signals
            else:
                self.signal += attenuated_signals

            self.traces.append(sig.trace)
            self.time = np.arange(0, duration, 1 / self.samplerate)

    def visualize(self, eoi=0, foi=0):
        import matplotlib.gridspec as gridspec
        import matplotlib.pyplot as plt
        import thunderfish.powerspectrum as ps

        # compute powerspectrum for each electrode and sum them up
        for i in range(rec.signal.shape[1]):
            spec, freqs, spectime = ps.spectrogram(
                data=rec.signal[:, i],
                ratetime=rec.samplerate,
                freq_resolution=0.5,
                overlap_frac=0.5,
            )
            if i == 0:
                summed_spec = spec
            else:
                summed_spec += spec

        fig = plt.figure(figsize=(12, 6))
        gs0 = gridspec.GridSpec(1, 2, figure=fig)
        gs00 = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs0[0], height_ratios=[1, 4]
        )
        ax1 = fig.add_subplot(gs00[0])
        ax2 = fig.add_subplot(gs00[1], sharex=ax1)
        ax3 = fig.add_subplot(gs0[1])
        ax3.set_aspect("equal")

        ax1.plot(rec.time, rec.signal[:, eoi], alpha=1, lw=1)
        ax1.set_xlim(rec.time.min(), rec.time.max())

        ax2.imshow(
            X=ps.decibel(summed_spec),
            aspect="auto",
            origin="lower",
            extent=(spectime.min(), spectime.max(), freqs.min(), freqs.max()),
            vmin=np.percentile(ps.decibel(summed_spec), 20),
            vmax=np.max(ps.decibel(summed_spec)),
        )
        ax2.set_ylim(400, 1100)
        ax2.set_xlim(spectime.min(), spectime.max())

        ax3.scatter(
            rec.ex, rec.ey, marker="o", color="white", alpha=1, lw=0, zorder=10
        )

        # plot positions of all fish
        for i in range(len(rec.x)):
            ax3.plot(rec.x[i], rec.y[i], alpha=0.8, lw=1)
        ax3.set_xlim(np.min(rec.x), np.max(rec.x))
        ax3.set_ylim(np.min(rec.y), np.max(rec.y))

        ax1.set_ylabel("amplitude [a.u.]")
        ax2.set_xlabel("time [s]")
        ax2.set_ylabel("frequency [Hz]")
        ax3.set_xlabel("x [m]")
        ax3.set_ylabel("y [m]")
        plt.show()


if __name__ == "__main__":
    rec = Recording(
        fishcount=4, duration=600, grid_shape=(4, 4), electrode_spacing=0.5
    )
    rec.visualize()

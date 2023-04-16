#!/usr/bin/env python3

import argparse
import gc

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

from models.modelhandling import ChirpNet, ChirpNet2, load_model
from utils.datahandling import find_on_time, resize_image
from utils.filehandling import ConfLoader, NumpyLoader
from utils.logger import make_logger
from utils.plotstyle import PlotStyle

logger = make_logger(__name__)
conf = ConfLoader("config.yml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ps = PlotStyle()


def group_close_chirps(chirps, time_tolerance=0.02):
    """
    Group close chirps into one chirp
    """
    grouped_chirps = []
    group = []
    for i, chirp in enumerate(chirps):
        if i == 0:
            group.append(chirp)
        else:
            if (chirp[0] - group[-1][0]) < time_tolerance:
                group.append(chirp)
            else:
                grouped_chirps.append(group)
                group = [chirp]
    if group:
        grouped_chirps.append(group)

    return grouped_chirps


def select_highest_prob_chirp(grouped_chirps):
    """
    Select the highest probability chirp from each group
    """

    best_chirps = []
    for i, group in enumerate(grouped_chirps):
        if len(group) > 1:
            probs = [chirp[2] for chirp in group]
            highest_prob = np.argmax(probs)
            best_chirps.append(group[highest_prob])
        else:
            best_chirps.append(group[0])
    return best_chirps


class Detector:
    def __init__(self, modelpath, dataset, mode):
        assert mode in [
            "memory",
            "disk",
        ], "Mode must be either 'memory' or 'disk'"
        logger.info("Initializing detector...")

        self.mode = mode
        self.model = load_model(modelpath, ChirpNet)
        self.data = dataset
        self.samplerate = conf.samplerate
        self.fill_samplerate = 1 / np.mean(np.diff(self.data.fill_times))
        self.freq_pad = conf.freq_pad
        self.time_pad = conf.time_pad
        self.window_size = int(conf.time_pad * 2 * self.fill_samplerate)
        self.stride = int(conf.stride * self.fill_samplerate)
        self.chirps = None

        if (self.data.times[-1] // 600 != 0) and (self.mode == "memory"):
            logger.warning(
                "It is recommended to process recordings longer than 10 minutes using the 'disk' mode"
            )

        if self.window_size % 2 == 0:
            self.window_size += 1
            logger.info(f"Time padding is not odd. Adding one.")

        if self.stride % 2 == 0:
            self.stride += 1
            logger.info(f"Stride is not odd. Adding one.")

    def classify_single(self, img):
        with torch.no_grad():
            img = torch.from_numpy(img).to(device)
            outputs = self.model(img)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, dim=1)
        probs = probs.cpu().numpy()[0][0]
        preds = preds.cpu().numpy()[0]
        return probs, preds

    def detect(self, plot=False):
        logger.info("Detecting...")
        if self.mode == "memory":
            self._detect_memory(plot)
        else:
            self._detect_disk(plot)

    def _detect_memory(self, plot):
        logger.info("Processing in memory...")

        first_index = 0
        last_index = self.data.fill_times.shape[0]
        window_start_indices = np.arange(
            first_index, last_index - self.window_size, self.stride, dtype=int
        )

        detected_chirps = []

        iter = 0
        for track_id in np.unique(self.data.ident_v):
            logger.info(f"Processing track {track_id}...")
            track = self.data.fund_v[self.data.ident_v == track_id]
            time = self.data.times[
                self.data.idx_v[self.data.ident_v == track_id]
            ]

            predicted_labels = []
            predicted_probs = []
            center_t = []
            center_f = []

            for window_start_index in window_start_indices:
                # Make index were current window will end
                window_end_index = window_start_index + self.window_size

                # Get the current frequency from the track
                center_idx = int(
                    window_start_index + np.floor(self.window_size / 2) + 1
                )
                window_center_t = self.data.fill_times[center_idx]
                track_index = find_on_time(time, window_center_t)
                center_freq = track[track_index]

                # From the track frequency compute the frequency
                # boundaries

                freq_min = center_freq + self.freq_pad[0]
                freq_max = center_freq + self.freq_pad[1]

                # Find these values on the frequency axis of the spectrogram
                freq_min_index = find_on_time(self.data.fill_freqs, freq_min)
                freq_max_index = find_on_time(self.data.fill_freqs, freq_max)

                # Using window start, stop and feeq lims, extract snippet from spec
                snippet = self.data.fill_spec[
                    freq_min_index:freq_max_index,
                    window_start_index:window_end_index,
                ]
                snippet = (snippet - np.min(snippet)) / (
                    np.max(snippet) - np.min(snippet)
                )
                snippet = resize_image(snippet, conf.img_size_px)
                snippet = np.expand_dims(snippet, axis=0)
                snippet = np.asarray([snippet]).astype(np.float32)
                prob, label = self.classify_single(snippet)

                # Append snippet to list
                predicted_labels.append(label)
                predicted_probs.append(prob)
                center_t.append(window_center_t)
                center_f.append(center_freq)

                iter += 1
                if not plot:
                    continue

                fig, ax = plt.subplots(1, 1, figsize=(24 * ps.cm, 12 * ps.cm))
                ax.imshow(
                    self.data.fill_spec,
                    aspect="auto",
                    origin="lower",
                    extent=[
                        self.data.fill_times[0],
                        self.data.fill_times[-1],
                        self.data.fill_freqs[0],
                        self.data.fill_freqs[-1],
                    ],
                    cmap="magma",
                    vmin=np.min(self.data.fill_spec) * 0.6,
                    vmax=np.max(self.data.fill_spec),
                    zorder=-100,
                    interpolation="gaussian",
                )
                # Create a Rectangle patch
                startx = self.data.fill_times[window_start_index]
                stopx = self.data.fill_times[window_end_index]
                starty = self.data.fill_freqs[freq_min_index]
                stopy = self.data.fill_freqs[freq_max_index]

                if label == 1:
                    patchc = ps.white
                else:
                    patchc = ps.maroon

                rect = Rectangle(
                    (startx, starty),
                    self.data.fill_times[window_end_index]
                    - self.data.fill_times[window_start_index],
                    self.data.fill_freqs[freq_max_index]
                    - self.data.fill_freqs[freq_min_index],
                    linewidth=2,
                    facecolor="none",
                    edgecolor=patchc,
                )

                # Add the patch to the Axes
                ax.add_patch(rect)

                # Add the chirpprob
                ax.text(
                    (startx + stopx) / 2,
                    stopy + 50,
                    f"{prob:.2f}",
                    color=ps.white,
                    fontsize=14,
                    horizontalalignment="center",
                    verticalalignment="center",
                )

                # Plot the track
                ax.plot(self.data.times, track, linewidth=1, color=ps.black)

                # Plot the window center
                ax.plot(
                    [
                        self.data.fill_times[
                            window_start_index + self.window_size // 2
                        ]
                    ],
                    [center_freq],
                    marker="o",
                    color=ps.black,
                )

                # make limits nice
                ax.set_ylim(np.min(track) - 200, np.max(track) + 400)
                startxw = startx - 5
                stopxw = stopx + 5

                if startxw < 0:
                    stopxw = stopxw - startxw
                    startxw = 0
                if stopxw > self.data.fill_times[-1]:
                    startxw = startxw - (stopxw - self.data.fill_times[-1])
                    stopxw = self.data.fill_times[-1]
                ax.set_xlim(startxw, stopxw)
                ax.axis("off")
                plt.subplots_adjust(left=-0.01, right=1, top=1, bottom=0)
                plt.savefig(f"../anim/test_{iter-1}.png")

                # Clear the plot
                plt.cla()
                plt.clf()
                plt.close("all")
                plt.close(fig)
                gc.collect()

            predicted_labels = np.asarray(predicted_labels)
            predicted_probs = np.asarray(predicted_probs)
            center_t = np.asarray(center_t)
            center_f = np.asarray(center_f)

            # detect the peaks in the probabilities
            # peaks of probabilities are chirps

            peaks, _ = find_peaks(predicted_probs, height=0.5)
            logger.info(f"ConvNet found {len(peaks)} chirps ")

            peaktimes = center_t[peaks]
            peakfreqs = center_f[peaks]
            peakid = np.repeat(int(track_id), len(peaktimes))
            peakprobs = predicted_probs[peaks]

            # put time, freq and id in list of touples
            chirps = list(zip(peaktimes, peakfreqs, peakprobs, peakid))

            # each chirp is now a touple with (time, freq, id, prob)
            detected_chirps.append(chirps)

        logger.info("Sorting detected chirps ...")

        # flatten the list of lists to a single list with all the chirps
        detected_chirps = np.concatenate(detected_chirps)

        # Sort chirps by time they were detected at
        detected_chirps = detected_chirps[detected_chirps[:, 0].argsort()]

        # group chirps that are close by in time to find the ones
        # that were detected twice on seperate fish

        grouped_chirps = group_close_chirps(detected_chirps, 0.02)

        # go through the close chirp and find the most probable one
        # for now just use the ConvNets prediction probability
        # this can be much more fancy, perhaps by going through
        # all duplicates and finding the stronges dip in the baseline
        # envelope. E.g. inverting the masked spectrogram for a single
        # fish would give a nice envelope. Bandpass filtering the envelope
        # also works but is not as robust as the spectrogram inversion (in
        # the case of strong fluctuations of the baseline e.g. during
        # a rise.)

        self.chirps = select_highest_prob_chirp(grouped_chirps)

    def _detect_disk(self, plot):
        logger.info("This function is not yet implemented. Aborting ...")

    def plot(self):
        d = self.data  # <----- Quick fix, remove this!!!
        # correct_chirps = np.load(
        #     conf.testing_data_path + "/correct_chirp_times.npy"
        # )
        # correct_chirp_ids = np.load(
        #     conf.testing_data_path + "/correct_chirp_time_ids.npy"
        # )

        fig, ax = plt.subplots(
            figsize=(24 * ps.cm, 12 * ps.cm), constrained_layout=True
        )
        ax.imshow(
            d.fill_spec,
            aspect="auto",
            origin="lower",
            extent=[
                d.fill_times[0],
                d.fill_times[-1],
                d.fill_freqs[0],
                d.fill_freqs[-1],
            ],
            zorder=-20,
            interpolation="gaussian",
        )

        for track_id in np.unique(d.ident_v):
            track_id = int(track_id)
            track = d.fund_v[d.ident_v == track_id]
            time = d.times[d.idx_v[d.ident_v == track_id]]

            ax.plot(time, track, linewidth=1, zorder=-10, color=ps.black)

        for chirp in self.chirps:
            t, f = chirp[0], chirp[1]
            ax.scatter(
                t,
                f,
                s=20,
                marker="o",
                color=ps.black,
                edgecolor=ps.black,
                zorder=10,
            )

        ax.set_ylim(np.min(d.fund_v - 100), np.max(d.fund_v + 300))
        ax.set_xlim(np.min(d.fill_times), np.max(d.fill_times))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")

        plt.savefig("../assets/detection.png")
        plt.show()


def interface():
    parser = argparse.ArgumentParser(
        description="Detects chirps on spectrograms."
    )
    parser.add_argument(
        "--path",
        type=str,
        default=conf.testing_data_path,
        help="Path to the dataset to use for detection",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="memory",
        help="Mode to use for detection. Can be either 'memory' or 'disk'. Defaults to 'memory'.",
    )
    args = parser.parse_args()
    return args


def main():
    args = interface()
    d = NumpyLoader(args.path)
    modelpath = conf.save_dir
    det = Detector(modelpath, d, args.mode)
    det.detect(plot=False)
    det.plot()


if __name__ == "__main__":
    main()

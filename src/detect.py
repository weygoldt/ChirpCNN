#!/usr/bin/env python3

import argparse
import gc
import pathlib

import matplotlib.pyplot as plt
import nixio as nio
import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks

from models.modelhandling import ChirpNet, ChirpNet2, load_model
from utils.datahandling import find_on_time, resize_image
from utils.filehandling import ConfLoader, load_data
from utils.logger import make_logger
from utils.plotstyle import PlotStyle
from utils.spectrogram import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    sint,
    spectrogram,
)

logger = make_logger(__name__)
conf = ConfLoader("config.yml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChirpNet
ps = PlotStyle()


def cluster_peaks(arr, thresh=0.5):
    """Clusters peaks of probabilitis between 0 and 1.
    Returns a list of lists where each list contains the indices of the
    all values belonging to a peak i.e. a cluster.

    Parameters
    ----------
    arr : np.ndarray
        Array of probabilities between 0 and 1.
    thresh : float, optional
        All values below are not peaks, by default 0.5

    Returns
    -------
    np.array(np.array(int))
        Each subarray contains the indices of the values belonging to a peak.
    """
    clusters = []
    cluster = []
    for i, val in enumerate(arr):
        # do nothing or append prev cluste if val is below threshold
        # then clear the current cluster
        if val <= thresh:
            if len(cluster) > 0:
                clusters.append(cluster)
                cluster = []
            continue

        # if larger than thresh
        # if first value in array, append to cluster
        # since there is no previous value to compare to
        if i == 0:
            cluster.append(i)

        # if this is the last value then there is no future value
        # to compare to so append to cluster
        elif i == len(arr) - 1:
            cluster.append(i)
            clusters.append(cluster)

        # if were at a trough then the diff between the current value and
        # the previous value will be negative and the diff between the
        # future value and the current value will be positive
        elif val - arr[i - 1] < 0 and arr[i + 1] - val > 0:
            cluster.append(i)
            clusters.append(cluster)
            cluster = []
            cluster.append(i)

        # if this is not the first value or the last value or a trough
        # then append to cluster
        else:
            cluster.append(i)

    return clusters


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


def minmaxnorm(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


def classify(model, img):
    with torch.no_grad():
        img = torch.from_numpy(img).to(device)
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, dim=1)
    probs = probs.cpu().numpy()[0][0]
    preds = preds.cpu().numpy()[0]
    return probs, preds


def detect_chirps(
    model,
    stride,
    window_size,
    spec,
    spec_freqs,
    spec_times,
    track_freqs,
    track_times,
    track_indices,
    track_idents,
):
    window_starts = np.arange(0, len(spec_times) - 1, stride, dtype=int)
    detect_chirps = []
    iter = 0

    for track_id in np.unique(track_idents):
        logger.info(f"Detecting chirps for track {track_id}")
        track = track_freqs[track_idents == track_id]
        time = track_times[track_indices[track_idents == track_id]]

        pred_labels = []
        pred_probs = []
        center_times = []
        center_freqs = []

        for i, window_start in enumerate(window_starts):
            # time axis indices
            min_time_index = window_start
            max_time_index = window_start + window_size
            center_time_index = int(window_start + window_size / 2)
            center_time = spec_times[center_time_index]

            # frequency axis indices
            window_center_track = find_on_time(time, center_time, False)
            window_center_freq = track[window_center_track]
            min_freq = window_center_freq - conf.freq_pad[0]
            max_freq = window_center_freq + conf.freq_pad[1]
            min_freq_idx = find_on_time(spec_freqs, min_freq)
            max_freq_idx = find_on_time(spec_freqs, max_freq)

            # get window area
            snippet = spec[
                min_freq_idx:max_freq_idx, min_time_index:max_time_index
            ]

            # normalize snippet
            snippet = minmaxnorm(snippet)

            # rezise to square
            snippet = resize_image(snippet, conf.image_size)

            # add a channel
            snippet = np.expand_dims(snippet, axis=0)

            # put it into an array
            snippet = np.asarray([snippet], dtype=np.float32)

            # predict the label and probability of the snippet
            prob, label = classify(model, snippet)

            # save the predictions and the center time and frequency
            pred_labels.append(label)
            pred_probs.append(prob)
            center_times.append(center_time)
            center_freqs.append(window_center_freq)
            iter += 1

        # convert to numpy arrays
        pred_labels = np.asarray(pred_labels)
        pred_probs = np.asarray(pred_probs)
        center_times = np.asarray(center_times)
        center_freqs = np.asarray(center_freqs)

        # get chirp clusters from the predictions
        cluster_indices = cluster_peaks(pred_probs, 0.5)

        # compute the weighted average of the center times and frequencies
        weighted_times = []
        weighted_freqs = []
        max_probs = []
        for cluster in cluster_indices:
            probs = pred_probs[cluster]
            times = center_times[cluster]
            freqs = center_freqs[cluster]
            weighted_freqs.append(np.average(freqs, weights=probs))
            weighted_times.append(np.average(times, weights=probs))
            max_probs.append(np.max(probs))

        # convert to numpy arrays
        weighted_times = np.asarray(weighted_times)
        weighted_freqs = np.asarray(weighted_freqs)
        max_probs = np.asarray(max_probs)
        chirp_id = np.repeat(track_id, len(weighted_times))

        logger.info(f"Found {len(weighted_times)} chirps")

        # put into touples for comparison across tracks
        current_chirps = list(
            zip(weighted_times, weighted_freqs, max_probs, chirp_id)
        )

        # append to the list of chirps
        detect_chirps.extend(current_chirps)

    logger.info("Sorting chirps...")

    # sort the chirps by time
    detected_chirps = np.asarray(detect_chirps)
    detected_chirps = detected_chirps[detected_chirps[:, 0].argsort()]

    # now group all chirps that are close in time and frequency
    grouped_chirps = group_close_chirps(detected_chirps, conf.min_chirp_dt)

    # now find the best chirp in each group
    chirps = select_highest_prob_chirp(grouped_chirps)

    return chirps


def extract_window(data, start, stop, samplerate):
    # crop the raw data
    raw = data.raw[start:stop, :]

    # time to snip the track data
    start_t = start / samplerate
    stop_t = stop / samplerate

    tracks = []
    indices = []
    idents = []
    for track_id in np.unique(data.track_idents):
        # make array for each track
        track = data.track_freqs[data.track_idents == track_id]
        time = data.track_times[
            data.track_indices[data.track_idents == track_id]
        ]
        index = data.track_indices[data.track_idents == track_id]

        # snip the track
        track = track[(time >= start_t) & (time <= stop_t)]
        index = index[(time >= start_t) & (time <= stop_t)]
        ident = np.repeat(track_id, len(track))

        # append to the list
        tracks.append(track)
        indices.append(index)
        idents.append(ident)

    # convert to numpy arrays
    tracks = np.concatenate(tracks)
    indices = np.concatenate(indices)
    indices -= indices[0]
    idents = np.concatenate(idents)
    time = data.track_times[
        (data.track_times >= start_t) & (data.track_times <= stop_t)
    ]
    return raw, tracks, idents, indices, time


class Detector:
    def __init__(self, modelpath, dataset):
        logger.info("Initializing detector...")

        # how many seconds of signal to process at a time
        self.buffersize = conf.buffersize

        # sampling rate of the raw signal
        self.samplingrate = conf.samplerate

        # how many electrodes were used
        self.n_electrodes = conf.n_electrodes

        # how many samples to shift the spectrogram window
        self.chunksize = conf.samplerate * conf.buffersize

        # frequency resolution of the spectrogram in nfft
        self.nfft = freqres_to_nfft(conf.frequency_resolution, conf.samplerate)

        # to what fraction the fft windows should overlap in indices
        self.hop_len = overlap_to_hoplen(conf.overlap_fraction, self.nfft)

        # how much overlap between individual spectrogram windows
        # to compensate for edge effects
        self.spectrogram_overlap = conf.spectrogram_overlap * conf.samplerate

        # load the dataset
        self.data = dataset

        # create parameters for the detector
        spec_samplerate = conf.sample_rate / self.hop_len
        window_size = int(conf.time_pad * 2 * spec_samplerate)
        if window_size % 2 == 0:
            window_size += 1
        stride = int(conf.stride * spec_samplerate)
        if stride % 2 == 0:
            stride += 1

        self.detection_parameters = {
            "model": load_model(modelpath, model),
            "stride": stride,
            "window_size": window_size,
        }

    def detect(self, plot=False):
        # number of chunks needed to process the whole dataset
        n_chunks = np.ceil(self.data.raw.shape[0] / self.chunksize).astype(int)

        # compute spectrogram parameters
        # nfft = freqres_to_nfft(self.frequency_resolution, self.samplingrate)
        # hop_length = overlap_to_hoplen(self.nfft_overlap, nfft)

        chirps = []
        timetracker = 0
        for i in range(n_chunks):
            logger.info(f"Processing chunk {i + 1} of {n_chunks}...")

            # get start and stop indices for the current chunk
            # including some overlap to compensate for edge effects
            # this diffrers for the first and last chunk
            if i == 0:
                idx1 = sint(i * self.chunksize)
                idx2 = sint((i + 1) * self.chunksize + self.spectrogram_overlap)
            if i == n_chunks - 1:
                idx1 = sint(i * self.chunksize - self.spectrogram_overlap)
                idx2 = sint((i + 1) * self.chunksize)
            else:
                idx1 = sint(i * self.chunksize - self.spectrogram_overlap)
                idx2 = sint((i + 1) * self.chunksize + self.spectrogram_overlap)

            # extract all data arrays for that window
            chunk, freqs, idents, indices, times = extract_window(
                self.data, idx1, idx2, self.samplingrate
            )

            # I do not understand anymore why this works but it worked
            # so lets leave it until I understand it
            spec_padding = int(self.spectrogram_overlap // self.hop_length)

            # compute the spectrogram for all electrodes
            for el in range(self.n_electrodes):
                chunk_spec, spec_freqs, spec_times = spectrogram(
                    chunk[:, el],
                    self.samplingrate,
                    nfft=self.nfft,
                    noverlap=self.nfft_overlap,
                )

                # sum spectrogram over all electrodes
                if el == 0:
                    spec = chunk_spec
                else:
                    spec += chunk_spec

            # normalize spectrogram by the number of electrodes
            spec /= self.n_electrodes

            # convert the spectrogram to dB
            spec = decibel(spec)

            # add timetracker to times and update timetracker
            spec_times += timetracker
            timetracker += spec_times[-1]

            # make a detection data dict
            detection_data = {
                "spec": spec,
                "spec_freqs": spec_freqs,
                "spec_times": spec_times,
                "track_freqs": freqs,
                "track_times": times,
                "track_idents": idents,
                "track_indices": indices,
            }

            # detect the chirps for the current chunk
            chunk_chirps = detect_chirps(
                **detection_data, **self.detection_parameters
            )

            # add the detected chirps to the list of all chirps
            chirps.extend(chunk_chirps)

        # reformat the detected chirps
        chirps = np.array(chirps)
        chirp_times = chirps[:, 0]
        chirp_ids = chirps[:, -1]


#         logger.info("Detecting...")

#         first_index = 0
#         last_index = self.data.spec_times.shape[0]
#         window_start_indices = np.arange(
#             first_index, last_index - self.window_size, self.stride, dtype=int
#         )

#         detected_chirps = []

#         iter = 0
#         for track_id in np.unique(self.data.track_idents):
#             logger.info(f"Processing track {track_id}...")
#             track = self.data.track_freqs[:][
#                 self.data.track_idents[:] == track_id
#             ]
#             time = self.data.track_times[:][
#                 self.data.track_indices[:][
#                     self.data.track_idents[:] == track_id
#                 ]
#             ]

#             predicted_labels = []
#             predicted_probs = []
#             center_t = []
#             center_f = []

#             for window_start_index in window_start_indices:
#                 # Make index were current window will end
#                 window_end_index = window_start_index + self.window_size

#                 # Get the current frequency from the track
#                 center_idx = int(
#                     window_start_index + np.floor(self.window_size / 2) + 1
#                 )
#                 window_center_t = self.data.spec_times[:][center_idx]
#                 track_index = find_on_time(time, window_center_t, limit=False)
#                 center_freq = track[track_index]

#                 # From the track frequency compute the frequency
#                 # boundaries

#                 freq_min = center_freq + self.freq_pad[0]
#                 freq_max = center_freq + self.freq_pad[1]

#                 # Find these values on the frequency axis of the spectrogram
#                 freq_min_index = find_on_time(self.data.spec_freqs[:], freq_min)
#                 freq_max_index = find_on_time(self.data.spec_freqs[:], freq_max)

#                 # Using window start, stop and feeq lims, extract snippet from spec
#                 snippet = self.data.spec[
#                     freq_min_index:freq_max_index,
#                     window_start_index:window_end_index,
#                 ]

#                 snippet = (snippet - np.min(snippet)) / (
#                     np.max(snippet) - np.min(snippet)
#                 )

#                 snippet = resize_image(snippet, conf.img_size_px)
#                 snippet = np.expand_dims(snippet, axis=0)
#                 snippet = np.asarray([snippet]).astype(np.float32)
#                 prob, label = self._classify_single(snippet)

#                 # fig, ax = plt.subplots()
#                 # ax.imshow(snippet[0][0], origin="lower")
#                 # ax.text(
#                 #     0.5,
#                 #     0.5,
#                 #     f"{prob:.2f}",
#                 #     horizontalalignment="center",
#                 #     verticalalignment="center",
#                 #     transform=ax.transAxes,
#                 #     color="white",
#                 #     fontsize=28,
#                 # )
#                 # ax.axis("off")
#                 # plt.savefig(
#                 #     f"../anim/{iter:05d}.png", dpi=300, bbox_inches="tight"
#                 # )
#                 # # Clear the plot
#                 # plt.cla()
#                 # plt.clf()
#                 # plt.close("all")
#                 # plt.close(fig)
#                 # gc.collect()

#                 # Append snippet to list
#                 predicted_labels.append(label)
#                 predicted_probs.append(prob)
#                 center_t.append(window_center_t)
#                 center_f.append(center_freq)

#                 iter += 1
#                 if not plot:
#                     continue

#                 # fig, ax = plt.subplots(1, 1, figsize=(24 * ps.cm, 12 * ps.cm))
#                 # ax.imshow(
#                 #     self.data.spec,
#                 #     aspect="auto",
#                 #     origin="lower",
#                 #     extent=[
#                 #         self.data.spec_times[:][0],
#                 #         self.data.spec_times[:][-1],
#                 #         self.data.spec_freqs[:][0],
#                 #         self.data.spec_freqs[:][-1],
#                 #     ],
#                 #     cmap="magma",
#                 #     zorder=-100,
#                 #     interpolation="gaussian",
#                 # )
#                 # # Create a Rectangle patch
#                 # startx = self.data.spec_times[:][window_start_index]
#                 # stopx = self.data.spec_times[:][window_end_index]
#                 # starty = self.data.spec_freqs[:][freq_min_index]
#                 # stopy = self.data.spec_freqs[:][freq_max_index]

#                 # if label == 1:
#                 #     patchc = ps.white
#                 # else:
#                 #     patchc = ps.maroon

#                 # rect = Rectangle(
#                 #     (startx, starty),
#                 #     self.data.spec_times[:][window_end_index]
#                 #     - self.data.spec_times[:][window_start_index],
#                 #     self.data.spec_freqs[:][freq_max_index]
#                 #     - self.data.spec_freqs[:][freq_min_index],
#                 #     linewidth=2,
#                 #     facecolor="none",
#                 #     edgecolor=patchc,
#                 # )

#                 # # Add the patch to the Axes
#                 # ax.add_patch(rect)

#                 # # Add the chirpprob
#                 # ax.text(
#                 #     (startx + stopx) / 2,
#                 #     stopy + 50,
#                 #     f"{prob:.2f}",
#                 #     color=ps.white,
#                 #     fontsize=14,
#                 #     horizontalalignment="center",
#                 #     verticalalignment="center",
#                 # )

#                 # # Plot the track
#                 # ax.plot(
#                 #     self.data.track_times[:], track, linewidth=1, color=ps.black
#                 # )

#                 # # Plot the window center
#                 # ax.plot(
#                 #     [
#                 #         self.data.spec_times[
#                 #             window_start_index + self.window_size // 2
#                 #         ]
#                 #     ],
#                 #     [center_freq],
#                 #     marker="o",
#                 #     color=ps.black,
#                 # )

#                 # # make limits nice
#                 # ax.set_ylim(np.min(track) - 200, np.max(track) + 400)
#                 # startxw = startx - 5
#                 # stopxw = stopx + 5

#                 # if startxw < 0:
#                 #     stopxw = stopxw - startxw
#                 #     startxw = 0
#                 # if stopxw > self.data.spec_times[-1]:
#                 #     startxw = startxw - (stopxw - self.data.spec_times[-1])
#                 #     stopxw = self.data.spec_times[-1]
#                 # ax.set_xlim(startxw, stopxw)
#                 # ax.axis("off")
#                 # plt.subplots_adjust(left=-0.01, right=1, top=1, bottom=0)
#                 # plt.savefig(f"../anim/test_{iter-1}.png")

#                 # # Clear the plot
#                 # plt.cla()
#                 # plt.clf()
#                 # plt.close("all")
#                 # plt.close(fig)
#                 # gc.collect()

#             predicted_labels = np.asarray(predicted_labels)
#             predicted_probs = np.asarray(predicted_probs)
#             center_t = np.asarray(center_t)
#             center_f = np.asarray(center_f)

#             # get clusters of chirps
#             cluster_indices = cluster_peaks(predicted_probs, 0.5)

#             # compute weigthed average time of chirps in cluster
#             weighted_fs = []
#             weighted_ts = []
#             maxprobs = []
#             for cluster in cluster_indices:
#                 probs = predicted_probs[cluster]
#                 times = center_t[cluster]
#                 freqs = center_f[cluster]
#                 weighted_ts.append(np.average(times, weights=probs))
#                 weighted_fs.append(np.average(freqs, weights=probs))
#                 maxprobs.append(np.max(probs))

#             weighted_ts = np.asarray(weighted_ts)
#             weighted_fs = np.asarray(weighted_fs)
#             maxprobs = np.asarray(maxprobs)
#             peakid = np.repeat(int(track_id), len(weighted_ts))

#             logger.info(f"ConvNet found {len(weighted_ts)} chirps ")

#             # put time, freq and id in list of touples
#             chirps = list(zip(weighted_ts, weighted_fs, maxprobs, peakid))

#             # each chirp is now a touple with (time, freq, id, prob)
#             detected_chirps.append(chirps)

#         logger.info("Sorting detected chirps ...")

#         # remove empty lists
#         detected_chirps = [x for x in detected_chirps if x]

#         # flatten the list of lists to a single list with all the chirps
#         detected_chirps = np.concatenate(detected_chirps)

#         # Sort chirps by time they were detected at
#         detected_chirps = detected_chirps[detected_chirps[:, 0].argsort()]

#         # group chirps that are close by in time to find the ones
#         # that were detected twice on seperate fish
#         grouped_chirps = group_close_chirps(detected_chirps, conf.min_chirp_dt)

#         # go through the close chirp and find the most probable one
#         # for now just use the ConvNets prediction probability
#         # this can be much more fancy, perhaps by going through
#         # all duplicates and finding the stronges dip in the baseline
#         # envelope. E.g. inverting the masked spectrogram for a single
#         # fish would give a nice envelope. Bandpass filtering the envelope
#         # also works but is not as robust as the spectrogram inversion (in
#         # the case of strong fluctuations of the baseline e.g. during
#         # a rise.)

#         self.chirps = select_highest_prob_chirp(grouped_chirps)

#         # save the chirps to a file
#         chirp_times = self.chirps[:, 0]
#         chirp_ids = self.chirps[:, -1]

#         # open nix file and put chirp times in there

#     def plot(self):
#         d = self.data  # <----- Quick fix, remove this!!!
#         # correct_chirps = np.load(
#         #     conf.testing_data_path + "/correct_chirp_times.npy"
#         # )
#         # correct_chirp_ids = np.load(
#         #     conf.testing_data_path + "/correct_chirp_time_ids.npy"
#         # )

#         fig, ax = plt.subplots(
#             figsize=(24 * ps.cm, 12 * ps.cm), constrained_layout=True
#         )

#         ax.imshow(
#             d.spec,
#             aspect="auto",
#             origin="lower",
#             extent=[
#                 d.spec_times[0],
#                 d.spec_times[-1],
#                 d.spec_freqs[0],
#                 d.spec_freqs[-1],
#             ],
#             zorder=-20,
#             interpolation="gaussian",
#         )

#         for track_id in np.unique(d.track_idents[:]):
#             track_id = int(track_id)
#             track = d.track_freqs[:][d.track_idents[:] == track_id]
#             time = d.track_times[d.track_indices[d.track_idents[:] == track_id]]
#             ax.plot(time, track, linewidth=1, zorder=-10, color=ps.black)
#             ax.text(time[10], track[0], str(track_id), color=ps.black)

#         for chirp in self.chirps:
#             t, f = chirp[0], chirp[1]
#             ax.scatter(
#                 t,
#                 f,
#                 s=20,
#                 marker="o",
#                 color=ps.black,
#                 edgecolor=ps.black,
#                 zorder=10,
#             )

#         ax.set_ylim(np.min(d.track_freqs - 100), np.max(d.track_freqs + 300))
#         ax.set_xlim(np.min(d.spec_times[:]), np.max(d.spec_times[:]))
#         ax.set_xlabel("Time [s]")
#         ax.set_ylabel("Frequency [Hz]")

#         plt.savefig("../assets/detection.png")
#         plt.show()


# def interface():
#     parser = argparse.ArgumentParser(
#         description="Detects chirps on spectrograms."
#     )
#     parser.add_argument(
#         "--path",
#         type=str,
#         default=conf.testing_data_path,
#         help="Path to the dataset to use for detection",
#     )
#     parser.add_argument(
#         "--mode",
#         type=str,
#         default="disk",
#         help="Mode to use for detection. Can be either 'memory' or 'disk'. Defaults to 'memory'.",
#     )
#     args = parser.parse_args()
#     return args


# def main():
#     args = interface()
#     d = load_data(pathlib.Path(args.path))
#     modelpath = conf.save_dir
#     det = Detector(modelpath, d, args.mode)
#     det.detect(plot=False)
#     det.plot()


# if __name__ == "__main__":
#     main()

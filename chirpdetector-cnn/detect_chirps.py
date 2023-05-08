#!/usr/bin/env python3

"""
Detect chirps on a single given dataset that containts a raw file and the wavetracker 
files. The raw file is used to compute a spectrogram and the wavetracker files are used
to slide the detector across the tracks of a single fish.
"""

import argparse
import pathlib
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython import embed
from models.audioclassifier import AudioClassifier
from models.modelhandling import check_device, load_model
from scipy.interpolate import interp1d
from utils.datahandling import (
    cluster_peaks,
    find_on_time,
    merge_duplicates,
    norm_tensor,
    resize_tensor_image,
)
from utils.filehandling import ConfLoader, DataSubset, load_data
from utils.filters import bandpass_filter
from utils.logger import make_logger
from utils.plotstyle import PlotStyle
from utils.spectrogram import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    sint,
    specshow,
    spectrogram,
)

logger = make_logger(__name__)
conf = ConfLoader("config.yml")
device = check_device()
model = AudioClassifier
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


def interpolate(data):
    """
    Interpolate the tracked frequencies to a regular sampling
    """
    track_freqs = []
    track_idents = []
    track_indices = []
    new_times = np.arange(
        data.track_times[0], data.track_times[-1], conf.stride
    )
    index_helper = np.arange(len(new_times))
    ids = np.unique(data.track_idents[~np.isnan(data.track_idents)])
    for track_id in ids:
        start_time = data.track_times[
            data.track_indices[data.track_idents == track_id][0]
        ]
        stop_time = data.track_times[
            data.track_indices[data.track_idents == track_id][-1]
        ]
        times_full = new_times[
            (new_times >= start_time) & (new_times <= stop_time)
        ]
        times_sampled = data.track_times[
            data.track_indices[data.track_idents == track_id]
        ]
        freqs_sampled = data.track_freqs[data.track_idents == track_id]
        f = interp1d(times_sampled, freqs_sampled, kind="cubic")
        freqs_interp = f(times_full)

        index_interp = index_helper[
            (new_times >= start_time) & (new_times <= stop_time)
        ]
        ident_interp = np.ones(len(freqs_interp)) * track_id

        track_idents.append(ident_interp)
        track_indices.append(index_interp)
        track_freqs.append(freqs_interp)

    track_idents = np.concatenate(track_idents)
    track_freqs = np.concatenate(track_freqs)
    track_indices = np.concatenate(track_indices)

    data.track_idents = track_idents
    data.track_indices = track_indices
    data.track_freqs = track_freqs
    data.track_times = new_times
    return data


def classify(model, img):
    """
    Classify a spectrogram image
    """
    with torch.no_grad():
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
    outer_iter,
    spec,
    spec_freqs,
    spec_times,
    track_freqs,
    track_times,
    track_indices,
    track_idents,
):
    window_starts = np.arange(
        0, len(spec_times) - window_size, stride, dtype=int
    )
    detect_chirps = []
    iter = 0

    # make blacklisted areas where vertical noise bands are too strong
    threshold = spec.std().cpu().numpy()
    noise_subset = spec[spec_freqs < 300]
    noise_profile = torch.mean(noise_subset, axis=0)
    noise_profile = noise_profile.cpu().numpy()
    noise_index = np.zeros_like(noise_profile, dtype=bool)
    noise_index[noise_profile > threshold] = True

    for track_id in np.unique(track_idents):
        logger.info(f"Detecting chirps for track {track_id}")
        track = track_freqs[track_idents == track_id]
        time = track_times[track_indices[track_idents == track_id]]

        # check if the track has data in this window
        if time[0] > spec_times[-1]:
            continue

        # make blacklisted areas where low amplitude is too low below
        # the frequency track

        pred_labels = []
        pred_probs = []
        center_times = []
        center_freqs = []

        """ I am trying to vectorize this for loop so I can scale instead of norm
        and it will run faster

        window_ranges = window_starts[:, np.newaxis] + np.arange(window_size)
        center_time_indices = window_ranges[:, int(window_size / 2)]
        center_times = spec_times[center_time_indices]

        window_center_track = np.asarray(
            [find_on_time(time, t, False) for t in center_times]
        )

        window_center_freq = track[window_center_track]

        window_center_freq_on_spec = np.asarray(
            [find_on_time(spec_freqs, f) for f in window_center_freq]
        )
        """

        for i, window_start in enumerate(window_starts):
            # check again if there is data in this window
            if time[0] > spec_times[window_start + window_size]:
                logger.info("First track time after window end, skipping")
                continue

            if time[-1] < spec_times[window_start]:
                logger.info("Last track time before window start, skipping")
                continue

            # if skip if current window touches a blacklisted noise band
            if True in noise_index[window_start : window_start + window_size]:
                logger.info("Noise band in window, skipping classification")
                continue

            # time axis indices
            min_time_index = window_start
            max_time_index = window_start + window_size
            center_time_index = int(window_start + window_size / 2)
            center_time = spec_times[center_time_index]

            # frequency axis indices
            window_center_track = find_on_time(time, center_time, True)
            if window_center_track is np.nan:
                print(np.min(time), np.max(time), center_time)
                window_center_track = find_on_time(time, center_time, False)

            window_center_freq = track[window_center_track]
            min_freq = window_center_freq + conf.freq_pad[0]
            max_freq = window_center_freq + conf.freq_pad[1]
            min_freq_idx = find_on_time(spec_freqs, min_freq)
            max_freq_idx = find_on_time(spec_freqs, max_freq)

            # get window area
            # spec is still a tensor
            snippet = spec[
                min_freq_idx:max_freq_idx, min_time_index:max_time_index
            ]

            if snippet.shape[-1] == 0:
                logger.info("Reached the end of the spectrogram, skipping")
                continue

            # this became redundant after scaling to 0 mean and 1 std
            # normalize snippet
            # still a tensor
            # snippet = norm_tensor(snippet)

            # rezise to square as tensor
            snippet = resize_tensor_image(snippet, conf.img_size_px)

            # convert to float32 because the model expects that
            snippet = snippet.to(torch.float32)

            # predict the label and probability of the snippet
            prob, label = classify(model, snippet)
            prob = 1 - prob

            # plot the snippet
            # fig, ax = plt.subplots()
            # ax.imshow(snippet[0][0].cpu().numpy(), origin="lower")
            # ax.text(0.5, 0.5, f"{prob:.2f}", color="white", fontsize=20)
            # plt.savefig(f"../anim_plots/{outer_iter}_{iter}.png")
            # plt.cla()
            # plt.clf()
            # plt.close("all")
            # plt.close(fig)

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
        cluster_indices = cluster_peaks(pred_probs, conf.min_chirp_prob)

        # compute the weighted average of the center times and frequencies
        # This is the first chirp sorting step!
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

        # logger.info("Removing chirps that are in low power areas ...")

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

    # if there are no chirps, return an empty list
    if len(detect_chirps) == 0:
        return [], noise_index
    # [], []

    detected_chirps = detected_chirps[detected_chirps[:, 0].argsort()]

    # now group all chirps that are close in time and frequency
    # the grouped chirps are simply many sublists containing multiple chirp
    # touples that are close in time
    grouped_chirps = group_close_chirps(detected_chirps, conf.min_chirp_dt)

    # Second chirp sorting step!
    # now find the best chirp in each group and return it
    # this means that no to chirps can occur simultaneously
    # in a conf.min_chirp_dt time window!
    chirps = select_highest_prob_chirp(grouped_chirps)

    logger.info(f"{len(chirps)} survived the sorting process")

    return chirps, noise_index


# noise_index, lowamp_index


class Detector:
    def __init__(self, modelpath, dataset):
        logger.info("Initializing detector...")

        # how many seconds of signal to process at a time
        self.buffersize = conf.buffersize

        # sampling rate of the raw signal
        self.samplingrate = conf.samplerate

        # how many electrodes were used
        # self.n_electrodes = conf.num_electrodes

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
        self.n_electrodes = self.data.n_electrodes

        # create parameters for the detector
        spec_samplerate = conf.samplerate / self.hop_len
        window_size = int(conf.time_pad * 2 * spec_samplerate)
        if window_size % 2 == 0:
            window_size += 1
        stride = int(conf.stride * spec_samplerate)
        if stride % 2 == 0:
            stride += 1
        self.passband = (
            np.min(self.data.track_freqs) - 100,
            np.max(self.data.track_freqs) + 100,
        )

        self.detection_parameters = {
            "model": load_model(modelpath, model),
            "stride": stride,
            "window_size": window_size,
        }

    def detect(self):
        # number of chunks needed to process the whole dataset
        n_chunks = np.ceil(self.data.raw.shape[0] / self.chunksize).astype(int)

        # TODO: Mask high amplitude vertical noise bands again

        chirps = []
        for i in range(n_chunks):
            logger.info(f"Processing chunk {i + 1} of {n_chunks}...")

            # get start and stop indices for the current chunk
            # including some overlap to compensate for edge effects
            # this diffrers for the first and last chunk

            if i == 0:
                idx1 = sint(i * self.chunksize)
                idx2 = sint((i + 1) * self.chunksize + self.spectrogram_overlap)
            elif i == n_chunks - 1:
                idx1 = sint(i * self.chunksize - self.spectrogram_overlap)
                idx2 = sint((i + 1) * self.chunksize)
            else:
                idx1 = sint(i * self.chunksize - self.spectrogram_overlap)
                idx2 = sint((i + 1) * self.chunksize + self.spectrogram_overlap)

            # check if start of the chunk is before the end of the data
            first_chunk_time = idx1 / self.samplingrate
            first_track_time = self.data.track_times[0]

            # compute the time of the spectrogram
            spec_times = (
                np.arange(idx1, idx2 + 1, self.hop_len) / self.samplingrate
            )
            spec_freqs = (
                np.arange(0, self.nfft / 2 + 1) * self.samplingrate / self.nfft
            )

            # skip if the chunk is before the first track
            if first_chunk_time < first_track_time:
                continue

            chunk = DataSubset(self.data, idx1, idx2)

            # check if the chunk has data
            if chunk.hasdata is False:
                logger.info("No data in chunk, skipping...")
                continue

            # compute the spectrogram for all electrodes
            for el in range(self.n_electrodes):
                # get the signal for the current electrode
                sig = chunk.raw[:, el]

                # compute the spectrogram for the current electrode
                chunk_spec, _, _ = spectrogram(
                    sig.copy(),
                    self.samplingrate,
                    nfft=self.nfft,
                    hop_length=self.hop_len,
                )

                # sum spectrogram over all electrodes
                # the spec is a tensor
                if el == 0:
                    spec = chunk_spec
                else:
                    spec += chunk_spec

            # normalize spectrogram by the number of electrodes
            # the spec is still a tensor
            spec /= self.n_electrodes

            # convert the spectrogram to dB
            # .. still a tensor
            spec = decibel(spec)

            # cut off everything outside the upper frequency limit
            # the spec is still a tensor
            spec = spec[spec_freqs <= conf.upper_spectrum_limit, :]
            spec_freqs = spec_freqs[spec_freqs <= conf.upper_spectrum_limit]

            # normalize the spectrogram to zero mean and unit variance
            # the spec is still a tensor
            spec = (spec - spec.mean()) / spec.std()

            # make a detection data dict
            # the spec is still a tensor!
            detection_data = {
                "outer_iter": i,
                "spec": spec,
                "spec_freqs": spec_freqs,
                "spec_times": spec_times,
                "track_freqs": chunk.track_freqs,
                "track_times": chunk.track_times,
                "track_idents": chunk.track_idents,
                "track_indices": chunk.track_indices,
            }

            # detect the chirps for the current chunk
            chunk_chirps, noise_index = detect_chirps(
                **detection_data, **self.detection_parameters
            )

            # add the detected chirps to the list of all chirps
            chirps.extend(chunk_chirps)

            # plot
            if len(chunk_chirps) > 0:
                fig, ax = plt.subplots(
                    1, 1, figsize=(20, 10), constrained_layout=True
                )
                specshow(
                    spec.cpu().numpy(),
                    spec_times,
                    spec_freqs,
                    ax,
                    aspect="auto",
                    origin="lower",
                )
                if len(noise_index) > 0:
                    ax.fill_between(
                        spec_times,
                        np.zeros(spec_times.shape),
                        noise_index * 2000,
                        color=ps.black,
                        alpha=0.6,
                    )
                for chirp in chunk_chirps:
                    ax.scatter(
                        chirp[0],
                        chirp[1],
                        facecolors="white",
                        edgecolors="black",
                        s=50,
                    )
                    ax.text(
                        chirp[0],
                        chirp[1] + 50,
                        np.round(chirp[2], 2),
                        fontsize=14,
                        color="white",
                        rotation="vertical",
                        va="bottom",
                        ha="center",
                    )
                ax.set_ylim(0, 1200)
                plt.savefig(
                    f"{conf.testing_data_path}/{str(self.data.path.name)}_{i}.png"
                )
                plt.cla()
                plt.clf()
                plt.close("all")
                plt.close(fig)

            del detection_data
            del spec

        # reformat the detected chirps
        chirps = np.array(chirps)
        if len(chirps) == 0:
            return None, None

        chirp_times = chirps[:, 0]
        chirp_ids = chirps[:, -1]

        # now iterate through the chirps of each track
        # and remove the duplicates
        new_chirps = []
        new_ids = []
        for track_id in np.unique(chirp_ids):
            track_chirps = chirp_times[chirp_ids == track_id]
            if len(track_chirps) == 0:
                continue

            track_chirps = np.sort(track_chirps)

            track_chirps = merge_duplicates(track_chirps, conf.min_chirp_dt / 2)
            new_chirps.extend(track_chirps)
            new_ids.extend([track_id] * len(track_chirps))

        # now we have a list of chirp times and a list of track ids
        # that are ready to save
        chirp_times = np.array(new_chirps)
        chirp_ids = np.array(new_ids)

        return chirp_times, chirp_ids


def interface():
    parser = argparse.ArgumentParser(
        description="Detects chirps on spectrograms."
    )
    parser.add_argument(
        "--path",
        "-p",
        type=str,
        default=conf.testing_data_path,
        help="Path to the dataset to use for detection",
    )
    args = parser.parse_args()
    return args


def main(path):
    datapath = pathlib.Path(path)
    data = load_data(datapath)
    modelpath = conf.save_dir

    # for trial of code
    # good chirp times for data: 2022-06-02-10_00
    # start = (3 * 60 * 60 + 6 * 60 + 20) * conf.samplerate
    # stop = start + 600 * conf.samplerate
    # data = DataSubset(data, start, stop)
    # data.track_times -= data.track_times[0]

    data = interpolate(data)
    det = Detector(modelpath, data)
    chirp_times, chirp_ids = det.detect()

    if chirp_times is None:
        logger.info("No chirps detected.")
        return

    logger.info(
        f"Detected {len(chirp_times)} chirps in {np.unique(chirp_ids)} fish."
    )
    logger.info(f"Saving detected chirps to {datapath}...")
    np.save(datapath / "chirp_times_cnn.npy", chirp_times)
    np.save(datapath / "chirp_ids_cnn.npy", chirp_ids)


if __name__ == "__main__":
    t0 = time.time()
    args = interface()
    main(args.path)
    t1 = time.time()
    print(f"Time elapsed: {t1 - t0:.2f} s")

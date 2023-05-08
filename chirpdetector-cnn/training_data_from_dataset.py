#!/usr/bin/env python3

"""
Augment real recordings with fake recordings to create a hybrid dataset
which should improve the performance of the model on real recordings.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from extract_training_data import main as extract_data
from fake_recording import (
    add_noise,
    add_vertical_noise_bands,
    make_chirp_times,
    make_chirps,
    make_eod,
    make_rises,
    motion_envelope,
    natural_scale,
    noise_envelope,
    zero_envelope,
)
from IPython import embed
from utils.filehandling import ConfLoader, DataSubset, load_data
from utils.logger import make_logger
from utils.plotstyle import PlotStyle
from utils.spectrogram import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    specshow,
    spectrogram,
)

conf = ConfLoader("config.yml")
logger = make_logger(__name__)
ps = PlotStyle()


@dataclass
class FakeRec:
    recording: np.ndarray
    track_freqs: np.ndarray
    track_idents: np.ndarray
    track_indices: np.ndarray
    track_times: np.ndarray
    chirp_times: np.ndarray
    chirp_ids: np.ndarray
    noise_times: np.ndarray


@dataclass
class HybridRec(FakeRec):
    spec: np.ndarray
    spec_times: np.ndarray
    spec_freqs: np.ndarray

    def save(self, path):
        np.save(path / "raw.npy", self.recording)
        np.save(path / "fill_spec.npy", self.spec)
        np.save(path / "fill_times.npy", self.spec_times)
        np.save(path / "fill_freqs.npy", self.spec_freqs)
        np.save(path / "fund_v.npy", self.track_freqs)
        np.save(path / "ident_v.npy", self.track_idents)
        np.save(path / "idx_v.npy", self.track_indices)
        np.save(path / "times.npy", self.track_times)
        np.save(path / "correct_chirp_times.npy", self.chirp_times)
        np.save(path / "correct_chirp_time_ids.npy", self.chirp_ids)
        np.save(path / "noise_times.npy", self.noise_times)
        logger.info(f"Saved hybrid recording to {path}")

    def plot(self, path):
        fig, ax = plt.subplots(figsize=(20, 10), constrained_layout=True)
        ylims = conf.eodfs[0] - 100, conf.eodfs[1] + 100
        specshow(
            self.spec,
            self.spec_times,
            self.spec_freqs,
            ax=ax,
            aspect="auto",
            origin="lower",
        )
        ax.scatter(
            self.noise_times,
            np.ones_like(self.noise_times) * ylims[0] + 100,
            color="white",
            marker="|",
        )

        for track_id in np.unique(self.track_idents):
            t = self.track_times[
                self.track_indices[self.track_idents == track_id]
            ]
            f = self.track_freqs[self.track_idents == track_id]
            ax.plot(t, f, "black", linewidth=1)
            chirpt = self.chirp_times[self.chirp_ids == track_id]
            ax.scatter(chirpt, np.ones_like(chirpt) * f[0], color="red", s=10)

        ax.set_ylim(*ylims)
        plt.savefig(path)
        logger.info(f"Saved hybrid recording plot to {path}")

        plt.cla()
        plt.clf()
        plt.close("all")
        plt.close(fig)


def get_free_freqs(freq_range, snippet):
    """
    Get frequencies that are not occupied by other tracks in the snippet
    """
    for track_id in np.unique(
        snippet.track_idents[~np.isnan(snippet.track_idents)]
    ):
        track = snippet.track_freqs[snippet.track_idents == track_id]
        freq_range = freq_range[
            (freq_range < track.min() - 5) | (freq_range > track.max() + 5)
        ]

    return freq_range


def add_vertical_noise_bands(time, free_time, recording):
    num_bands = np.random.randint(
        conf.vertical_noise_bands[0], conf.vertical_noise_bands[1]
    )
    band_widths = np.random.uniform(
        conf.chirp_durations[0],
        conf.chirp_durations[1],
        size=num_bands,
    )
    band_stds = np.random.uniform(
        conf.vertical_noise_band_stds[0],
        conf.vertical_noise_band_stds[1],
        size=num_bands,
    )
    band_starts = np.random.choice(free_time, size=num_bands, replace=False)

    band_centers = band_starts + band_widths / 2
    for i in range(len(band_starts)):
        start = band_starts[i]
        end = start + band_widths[i]
        std = band_stds[i]
        size = len(time[(time >= start) & (time < end)])

        # make noise
        noise = np.random.normal(0, std, size=size)

        # add amplitude modulated
        ampmod = np.linspace(0, 1, len(noise))[::-1]

        noise *= ampmod

        recording[(time >= start) & (time < end)] += noise

    return recording, band_centers


def to_spectrogram(recording):
    nfft = freqres_to_nfft(conf.frequency_resolution, conf.samplerate)
    hop_length = overlap_to_hoplen(conf.overlap_fraction, nfft)

    s, t, f = spectrogram(
        recording,
        conf.samplerate,
        nfft,
        hop_length,
        trycuda=False,
    )

    return s.numpy(), t, f


def get_stats(data):
    """
    Get mean and std of data
    """
    stats = {
        "mu": np.mean(data),
        "std": np.std(data),
    }
    return stats


def fake_fish(eodfs, duration, samplerate, stats):
    """
    Create fake fish with chirps, rises and noise for each supplied eodf
    """

    time = np.arange(0, duration, 1 / samplerate)
    traces = []
    trace_ids = []
    trace_indices = []
    chirp_times = []
    chirp_ids = []

    for track_id, eodf in enumerate(eodfs):
        # make freq and amp traces
        n_envelope = noise_envelope(time)
        m_envelope = motion_envelope(time)
        blacklist, z_envelope = zero_envelope(time)
        chirp_t = make_chirp_times(time[blacklist])
        chirp_trace, c_envelope = make_chirps(chirp_t)
        rise_trace = make_rises(time)

        # put it all together
        trace = chirp_trace + rise_trace + eodf
        trace = add_noise(trace)

        # update rise trace with eodf, we need it later
        rise_trace += eodf

        # combine the envelopes and add variation in amplitudes
        ampscale = np.random.uniform(conf.amp_scale[0], conf.amp_scale[1])
        envelope = n_envelope * m_envelope * c_envelope * z_envelope * ampscale

        # make the eod
        eod = make_eod(trace)

        # modulate the amplitude of the eod
        eod *= envelope

        # store the data
        traces.append(rise_trace)
        trace_ids.append(np.ones_like(rise_trace) * track_id)
        trace_indices.append(np.arange(len(rise_trace)))
        chirp_times.append(chirp_t)
        chirp_ids.append(np.ones_like(chirp_t) * track_id)

        if track_id == 0:
            recording = eod
        else:
            recording += eod

    # reshape the collected data lists to arrays
    traces = np.concatenate(traces)
    trace_ids = np.concatenate(trace_ids)
    trace_indices = np.concatenate(trace_indices)
    chirp_times = np.concatenate(chirp_times)
    chirp_ids = np.concatenate(chirp_ids)

    # get time outside chirps where we can add noise
    time_outside_chirps = time[~np.isin(time, chirp_times)]

    # add the vertical noise bands
    recording, noise_time = add_vertical_noise_bands(
        time, time_outside_chirps, recording
    )

    # scale recording to match real mean and std
    recording = natural_scale(recording, stats)

    fake = FakeRec(
        recording=recording,
        track_freqs=traces,
        track_idents=trace_ids,
        track_indices=trace_indices,
        track_times=time,
        chirp_times=chirp_times,
        chirp_ids=chirp_ids,
        noise_times=noise_time,
    )

    return fake


def sum_spectrogram(fake: FakeRec, samplerate: int) -> HybridRec:
    """
    Sum spectrograms of multiple signals
    """
    nfft = freqres_to_nfft(conf.frequency_resolution, samplerate)
    hoplen = overlap_to_hoplen(conf.overlap_fraction, nfft)

    for el in range(fake.recording.shape[1]):
        sig = fake.recording[:, el]
        s, t, f = spectrogram(
            sig.copy(), samplerate, nfft=nfft, hop_length=hoplen, trycuda=False
        )
        if el == 0:
            spec = s
        else:
            spec += s

    spec /= fake.recording.shape[1]
    spec = decibel(spec)
    spec = spec.cpu().numpy()

    hybrid = HybridRec(
        **fake.__dict__,
        spec=spec,
        spec_times=t,
        spec_freqs=f,
    )

    return hybrid


def crop_tracks(hybrid: HybridRec) -> HybridRec:
    traces_c, trace_ids_c, trace_idx_c = [], [], []
    start, stop = hybrid.spec_times[0], hybrid.spec_times[-1]

    for fish_id in np.unique(hybrid.track_idents):
        trace = hybrid.track_freqs[hybrid.track_idents == fish_id][
            (hybrid.track_times >= start) & (hybrid.track_times <= stop)
        ]
        traces_c.append(trace)
        trace_ids_c.append(np.ones_like(trace) * fish_id)
        trace_idx_c.append(np.arange(len(trace)))
    time_c = hybrid.track_times[
        (hybrid.track_times >= start) & (hybrid.track_times <= stop)
    ]
    track_freqs = np.concatenate(traces_c)
    track_idents = np.concatenate(trace_ids_c)
    track_indices = np.concatenate(trace_idx_c)
    track_times = time_c

    hybrid.track_freqs = track_freqs
    hybrid.track_idents = track_idents
    hybrid.track_indices = track_indices
    hybrid.track_times = track_times

    return hybrid


def parse_dataset(datapath):
    """
    Go through a single dataset and generate semi-synthetic ground truths
    for chirp times.
    """

    data = load_data(datapath)

    duration = conf.simulation_duration_rec

    # range of possible simulated fish eodfs
    freq_range = np.arange(conf.eodfs[0], conf.eodfs[1])

    # set up how many windows and pick randomly where to extract windows
    window_size = duration * data.samplerate
    start_index = window_size
    stop_index = data.raw.shape[0] - window_size
    max_n_windows = (stop_index - start_index) // window_size
    n_windows = conf.windows_per_recording

    if n_windows > max_n_windows:
        logger.warning(
            f"Requested {n_windows} windows per recording, but only {max_n_windows} are available."
        )
        n_windows = max_n_windows

    window_starts = np.random.randint(start_index, stop_index, n_windows)

    logger.info(f"Parsing recording {datapath}...")
    for i, start in enumerate(window_starts):
        logger.info(f"Processing window {i+1}/{n_windows}")

        stop = start + window_size
        snippet = DataSubset(data, start, stop)
        stats = get_stats(snippet.raw)

        # get parameters for fake fish
        free_freqs = get_free_freqs(freq_range, snippet)
        num_fish = np.random.randint(conf.num_fish[0], conf.num_fish[1])
        fake_eods = np.random.choice(free_freqs, num_fish, replace=False)
        fake = fake_fish(fake_eods, duration, data.samplerate, stats)

        # add the real recording onto the fake recording
        fake.recording = snippet.raw + fake.recording[:, np.newaxis]

        # compute the spectrogram of chimera
        hybrid = sum_spectrogram(fake, data.samplerate)

        # crop frequency tracks to match spectrogram time axis
        hybrid = crop_tracks(hybrid)

        # save and plot
        path = Path(conf.testing_data_path)
        path.mkdir(parents=True, exist_ok=True)
        hybrid.plot(path / f"hybrid_{i}.png")
        hybrid.save(path)

        # run the data extractor on the hybrid recording
        extract_data()

        del snippet
        del hybrid
        del fake


def main():
    testpath = Path(
        "/home/weygoldt/projects/chirpdetector/data/2022-06-02-10_00"
    )
    parse_dataset(testpath)


if __name__ == "__main__":
    main()

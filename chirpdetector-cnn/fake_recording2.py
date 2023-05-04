#!/usr/bin/env python3

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
from models.modelhandling import check_device
from simulations.fish_signal import chirps, rises, wavefish_eods
from utils.datahandling import find_on_time
from utils.filehandling import ConfLoader
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

device = check_device()


def noise_envelope(time):
    noise_std = np.random.uniform(conf.noise_stds[0], conf.noise_stds[1])
    n_envelope = np.random.normal(1, noise_std, size=len(time))
    return n_envelope


def motion_envelope(time):
    motion_envelope_f = np.random.uniform(
        conf.motion_envelope_f[0], conf.motion_envelope_f[1]
    )
    motion_envelope_a = np.random.uniform(
        conf.motion_envelope_a[0], conf.motion_envelope_a[1]
    )
    motion_envelope = (
        np.sin(2 * np.pi * motion_envelope_f * time) * motion_envelope_a
    )
    motion_envelope += 1 - np.max(motion_envelope)
    return motion_envelope


def zero_envelope(time):
    num_zeros = np.random.randint(conf.num_zeros[0], conf.num_zeros[1])
    zero_times = np.random.uniform(0, time[-1], size=num_zeros)
    zero_durations = np.random.uniform(
        conf.chirp_durations[0], conf.chirp_durations[1] * 100, size=num_zeros
    )
    _, zero_envelope = chirps(
        0,
        conf.samplerate,
        conf.simulation_duration_rec,
        zero_times,
        np.zeros_like(zero_times),
        zero_durations,
        np.random.uniform(
            conf.chirp_kurtoses[0], conf.chirp_kurtoses[1], size=num_zeros
        ),
        np.ones_like(zero_times),
    )
    blacklist = np.ones_like(time, dtype=bool)
    blacklist[zero_envelope < 1] = False

    return blacklist, zero_envelope


def make_chirp_times(valid_times):
    num_chirps = np.random.randint(conf.num_chirps[0], conf.num_chirps[1])
    chirp_times = np.random.choice(valid_times, size=num_chirps, replace=False)

    return chirp_times


def make_chirps(chirp_times):
    num_chirps = len(chirp_times)
    chirp_sizes = np.random.uniform(
        conf.chirp_sizes[0], conf.chirp_sizes[1], size=num_chirps
    )
    chirp_durations = np.random.uniform(
        conf.chirp_durations[0], conf.chirp_durations[1], size=num_chirps
    )
    chirp_kurtoses = np.random.uniform(
        conf.chirp_kurtoses[0], conf.chirp_kurtoses[1], size=num_chirps
    )
    chirp_contrasts = np.random.uniform(
        conf.chirp_contrasts[0], conf.chirp_contrasts[1], size=num_chirps
    )
    chirp_trace, chirp_envelope = chirps(
        0,
        conf.samplerate,
        conf.simulation_duration_rec,
        chirp_times,
        chirp_sizes,
        chirp_durations,
        chirp_kurtoses,
        chirp_contrasts,
    )

    return chirp_trace, chirp_envelope


def make_rises(time):
    num_rises = np.random.randint(conf.num_rises[0], conf.num_rises[1])
    rise_times = np.random.uniform(0, time[-1], size=num_rises)
    rise_sizes = np.random.uniform(
        conf.rise_sizes[0], conf.rise_sizes[1], size=num_rises
    )
    rise_rise_taus = np.random.uniform(
        conf.rise_rise_taus[0], conf.rise_rise_taus[1], size=num_rises
    )
    rise_decay_taus = np.random.uniform(
        conf.rise_decay_taus[0], conf.rise_decay_taus[1], size=num_rises
    )
    rise_trace = rises(
        0,
        conf.samplerate,
        conf.simulation_duration_rec,
        rise_times,
        rise_sizes,
        rise_rise_taus,
        rise_decay_taus,
    )

    return rise_trace


def make_eod(trace):
    eod = wavefish_eods(
        "Alepto",
        trace,
        conf.samplerate,
        conf.simulation_duration_rec,
        phase0=0,
        noise_std=0,
    )

    return eod


def add_noise(eod):
    noise_std = np.random.uniform(conf.noise_stds[0], conf.noise_stds[1])
    noise = np.random.normal(0, noise_std, size=len(eod))
    eod += noise

    return eod


def add_vertical_noise_bands(time, recording):
    num_bands = np.random.randint(
        conf.vertical_noise_bands[0], conf.vertical_noise_bands[1]
    )
    band_widths = np.random.uniform(
        conf.vertical_noise_band_widths[0],
        conf.vertical_noise_band_widths[1],
        size=num_bands,
    )
    band_stds = np.random.uniform(
        conf.vertical_noise_band_stds[0],
        conf.vertical_noise_band_stds[1],
        size=num_bands,
    )
    band_starts = np.random.choice(time, size=num_bands, replace=False)

    for i in range(len(band_starts)):
        start = band_starts[i]
        end = start + band_widths[i]
        std = band_stds[i]
        noise = np.random.normal(0, std, size=len(recording))
        recording[(time >= start) & (time < end)] += noise[
            (time >= start) & (time < end)
        ]

    return recording


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

    s = decibel(s, trycuda=False).numpy()

    return s, t, f


def fake_recording():
    logger.info("Generating fake recording")

    # generate recording parameters
    time = np.arange(0, conf.simulation_duration_rec, 1 / conf.samplerate)
    num_fish = np.random.randint(conf.num_fish[0], conf.num_fish[1], size=1)
    eodfs = np.random.randint(conf.eodfs[0], conf.eodfs[1], size=num_fish)

    # generate arrays to store the traces and the correct chirp times
    traces = []
    trace_ids = []
    correct_chirp_times = []
    correct_chirp_time_ids = []

    # loop over all fish and generate data for each fish
    for fish, eodf in enumerate(eodfs):
        # make noise envelope
        n_envelope = noise_envelope(time)

        # model motion envelope by a sine wave
        m_envelope = motion_envelope(time)

        # add random drops in amplitude
        blacklist, z_envelope = zero_envelope(time)

        # make chirps outside random drops in amplitude
        chirp_times = make_chirp_times(time[blacklist])
        chirp_trace, c_envelope = make_chirps(chirp_times)

        # make rise trace
        rise_trace = make_rises(time)

        # combine rise and chirp trace and shift up to eodf
        trace = chirp_trace + rise_trace + eodf

        # update rise trace with eodf, we need it later without the chirps
        rise_trace += eodf

        # combine noise, motion, zero and chirp envelopes
        envelope = n_envelope * m_envelope * z_envelope * c_envelope

        # make the eod
        eod = make_eod(trace)

        # modulate the amplitude of the eod
        eod *= envelope

        # store data
        traces.append(rise_trace)
        trace_ids.append(np.ones_like(rise_trace) * fish)
        correct_chirp_times.append(chirp_times)
        correct_chirp_time_ids.append(np.ones_like(chirp_times) * fish)

        # add noise to the first fish only because it sums up
        # then store subsequent fish without noise
        if fish == 0:
            recording = add_noise(eod)
        else:
            recording += add_noise(eod)

    # add these pesky vertical noise bands
    recording = add_vertical_noise_bands(time, recording)

    # reformat the tracks and chirp times
    traces = np.concatenate(traces)
    trace_ids = np.concatenate(trace_ids)
    correct_chirp_times = np.concatenate(correct_chirp_times)
    correct_chirp_time_ids = np.concatenate(correct_chirp_time_ids)

    # compute and plot spectrogram
    spec, spec_times, spec_freqs = to_spectrogram(recording)

    fig, ax = plt.subplots(figsize=(10, 5))
    specshow(spec, spec_times, spec_freqs, ax=ax, aspect="auto", origin="lower")

    for track_id in np.unique(correct_chirp_time_ids):
        f = traces[trace_ids == track_id]
        chirpt = correct_chirp_times[correct_chirp_time_ids == track_id]
        chirpf = np.asarray([f[find_on_time(time, t)] for t in chirpt])
        ax.scatter(chirpt, chirpf)
        ax.plot(time, f, color="black")

    ax.set_ylim(np.min(traces) - 100, np.max(traces) + 100)
    plt.show()

    # TODO: Add correct chirp amplitude range in mV
    # TODO: Improve vertical random noise bands, add ampmod
    # TODO: Add general background noise
    # TODO: Export the generated data


def main():
    fake_recording()


if __name__ == "__main__":
    main()

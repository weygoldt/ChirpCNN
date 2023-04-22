#!/usr/bin/env python3

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed

from simulations.fish_signal import chirps, rises, wavefish_eods
from utils.datahandling import find_on_time
from utils.filehandling import ConfLoader
from utils.logger import make_logger
from utils.plotstyle import PlotStyle
from utils.spectrogram import (
    decibel,
    freqres_to_nfft,
    overlap_to_hoplen,
    spectrogram,
)

conf = ConfLoader("config.yml")
logger = make_logger(__name__)
ps = PlotStyle()

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def main(mode):
    logger.info("Generating fake recording")

    nfft = freqres_to_nfft(conf.frequency_resolution, conf.samplerate)
    hop_length = overlap_to_hoplen(conf.overlap_fraction, nfft)

    time = np.arange(0, conf.simulation_duration_rec, 1 / conf.samplerate)

    num_fish = np.random.randint(conf.num_fish[0], conf.num_fish[1], size=1)
    eodfs = np.random.randint(conf.eodfs[0], conf.eodfs[1], size=num_fish)

    traces = []
    correct_chirp_times = []
    correct_chirp_time_ids = []

    for fish, eodf in enumerate(eodfs):
        num_chirps = np.random.randint(conf.num_chirps[0], conf.num_chirps[1])
        chirp_times = np.random.uniform(0, time[-1], size=num_chirps)
        chirp_sizes = np.random.uniform(
            conf.chirp_sizes[0], conf.chirp_sizes[1], size=num_chirps
        )
        chirp_durations = np.random.uniform(
            conf.chirp_durations[0],
            conf.chirp_durations[1],
            size=num_chirps,
        )
        chirp_kurtoses = np.random.uniform(
            conf.chirp_kurtoses[0], conf.chirp_kurtoses[1], size=num_chirps
        )
        chirp_contrasts = np.random.uniform(
            conf.chirp_contrasts[0],
            conf.chirp_contrasts[1],
            size=num_chirps,
        )
        chirp_trace, amplitude_modulation = chirps(
            0,
            conf.samplerate,
            conf.simulation_duration_rec,
            chirp_times,
            chirp_sizes,
            chirp_durations,
            chirp_kurtoses,
            chirp_contrasts,
        )

        num_rises = np.random.randint(conf.num_rises[0], conf.num_rises[1])
        rise_times = np.random.uniform(
            0, conf.simulation_duration_rec, size=num_rises
        )
        rise_sizes = np.random.uniform(
            conf.rise_sizes[0], conf.rise_sizes[1], size=num_rises
        )
        rise_rise_taus = np.random.uniform(
            conf.rise_rise_taus[0], conf.rise_rise_taus[1], size=num_rises
        )
        rise_decay_taus = np.random.uniform(
            conf.rise_decay_taus[0],
            conf.rise_decay_taus[1],
            size=num_rises,
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

        eod_trace = rise_trace + chirp_trace + eodf

        rise_trace += eodf
        traces.append(rise_trace)
        correct_chirp_times.append(chirp_times)
        correct_chirp_time_ids.append(np.ones_like(chirp_times) * fish)

        eod = wavefish_eods(
            "Alepto",
            eod_trace,
            conf.samplerate,
            conf.simulation_duration_rec,
            phase0=0,
            noise_std=0,
        )

        # modulate amplitude to simulate chirp amplitude decrease
        eod = eod * amplitude_modulation

        # modulate amplitude to simulate movement
        # this still needs to be implemented

        # add noise in one iter only to avoid adding too much noised
        if fish == 0:
            recording = eod
        else:
            recording += eod

    recording = recording / len(eodfs)

    # reshape to make electrode grid
    original_recording = recording
    for i in range(conf.num_electrodes):
        noise_std = np.random.uniform(conf.noise_stds[0], conf.noise_stds[1])
        noise = noise_std * np.random.randn(len(eod))
        noise_recording = original_recording + noise
        if i == 0:
            recording = noise_recording
        else:
            recording = np.vstack((recording, noise_recording))

    recording = recording.T
    outpath = pathlib.Path(conf.testing_data_path)
    outpath.mkdir(parents=True, exist_ok=True)

    if mode == "default":
        for e in range(np.shape(recording)[1]):
            spec, _, _ = spectrogram(
                recording[:, e],
                conf.samplerate,
                nfft,
                hop_length,
                trycuda=False,
            )

            if e == 0:
                spec = spec
            else:
                spec += spec

        spec = decibel(spec, trycuda=False).numpy() / conf.num_electrodes

        # get time and frequency axis
        spec_times = np.arange(0, spec.shape[1]) * hop_length / conf.samplerate
        frequencies = np.arange(0, spec.shape[0]) * conf.samplerate / nfft

        traces_cropped, trace_ids, trace_idx = [], [], []
        spec_min, spec_max = np.min(spec_times), np.max(spec_times)

        for fish, trace in enumerate(traces):
            traces_cropped.append(
                trace[(time >= spec_min) & (time <= spec_max)]
            )
            trace_ids.append(np.ones_like(traces_cropped[-1]) * fish)
            trace_idx.append(np.arange(len(traces_cropped[-1])))

        time_cropped = time[(time >= spec_min) & (time <= spec_max)]

        fund_v = np.concatenate(traces_cropped)
        ident_v = np.concatenate(trace_ids)
        idx_v = np.concatenate(trace_idx)
        times = time_cropped
        np.save(outpath / "fill_spec.npy", spec)
        np.save(outpath / "fill_freqs.npy", frequencies)
        np.save(outpath / "fill_times.npy", spec_times)
    else:
        trace_ids = [[i] * len(t) for i, t in enumerate(traces)]
        trace_idxs = [np.arange(len(t)) for t in traces]
        fund_v = np.concatenate(traces)
        ident_v = np.concatenate(trace_ids)
        times = time
        idx_v = np.concatenate(trace_idxs)
        raw = recording
        embed()
        np.save(outpath / "raw.npy", raw)

    np.save(outpath / "fund_v.npy", fund_v)
    np.save(outpath / "ident_v.npy", ident_v)
    np.save(outpath / "idx_v.npy", idx_v)
    np.save(outpath / "times.npy", times)
    np.save(
        outpath / "correct_chirp_times.npy", np.concatenate(correct_chirp_times)
    )
    np.save(
        outpath / "correct_chirp_time_ids.npy",
        np.concatenate(correct_chirp_time_ids),
    )


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        "-m",
        type=str,
        default="default",
    )
    args = parser.parse_args()
    assert args.mode in [
        "default",
        "test",
    ], "mode not recognized, use default or test"
    return args


if __name__ == "__main__":
    args = interface()
    main(args.mode)

    spectrogram = np.load(conf.testing_data_path + "/fill_spec.npy")
    frequencies = np.load(conf.testing_data_path + "/fill_freqs.npy")
    spec_times = np.load(conf.testing_data_path + "/fill_times.npy")
    traces = np.load(conf.testing_data_path + "/fund_v.npy")
    trace_ids = np.load(conf.testing_data_path + "/ident_v.npy")
    time = np.load(conf.testing_data_path + "/times.npy")
    correct_chirp_times = np.load(
        conf.testing_data_path + "/correct_chirp_times.npy"
    )
    correct_chirp_time_ids = np.load(
        conf.testing_data_path + "/correct_chirp_time_ids.npy"
    )

    fig, ax = plt.subplots(
        figsize=(24 * ps.cm, 10 * ps.cm),
        constrained_layout=True,
    )

    ax.imshow(
        spectrogram,
        aspect="auto",
        origin="lower",
        extent=[spec_times[0], spec_times[-1], frequencies[0], frequencies[-1]],
        interpolation="gaussian",
    )

    for trace_id in np.unique(trace_ids):
        ax.plot(time, traces[trace_ids == trace_id], color=ps.black)

        id_chirp_times = correct_chirp_times[correct_chirp_time_ids == trace_id]
        time_index = np.arange(len(time))
        freq_index = [find_on_time(time, x, False) for x in id_chirp_times]

        ax.plot(
            id_chirp_times,
            traces[trace_ids == trace_id][freq_index],
            "|",
            color=ps.black,
        )

    ax.set_xlim(spec_times[0], spec_times[-1])
    # ax.set_ylim(np.min(traces) - 100, np.max(traces) + 300)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    plt.savefig("../assets/chirps.png")
    plt.show()

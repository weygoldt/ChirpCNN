#!/usr/bin/env python3

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython import embed
from scipy.signal import resample
from torchaudio.transforms import AmplitudeToDB, Spectrogram

from numpy_to_nix import freqres_to_nfft, overlap_to_hoplen
from simulations.fish_signal import chirps, rises, wavefish_eods
from utils.datahandling import find_on_time
from utils.filehandling import ConfLoader
from utils.logger import make_logger
from utils.plotstyle import PlotStyle

conf = ConfLoader("config.yml")
logger = make_logger(__name__)
ps = PlotStyle()

device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"


def main():
    logger.info("Generating fake recording")

    nfft = freqres_to_nfft(conf.frequency_resolution, conf.samplerate)
    hop_length = overlap_to_hoplen(conf.overlap_fraction, nfft)

    spectrogram_of = Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        normalized=True,
    ).to(device)
    in_decibel = AmplitudeToDB(stype="power", top_db=80).to(device)

    time = np.arange(0, conf.simulation_duration_rec, 1 / conf.samplerate)

    eodfs = np.random.randint(conf.eodfs[0], conf.eodfs[1], size=conf.num_fish)

    traces = []
    correct_chirp_times = []
    correct_chirp_time_ids = []

    for fish, eodf in enumerate(eodfs):
        chirp_times = np.random.uniform(0, time[-1], size=conf.num_chirps)
        chirp_sizes = np.random.uniform(
            conf.chirp_sizes[0], conf.chirp_sizes[1], size=conf.num_chirps
        )
        chirp_durations = np.random.uniform(
            conf.chirp_durations[0],
            conf.chirp_durations[1],
            size=conf.num_chirps,
        )
        chirp_kurtoses = np.random.uniform(
            conf.chirp_kurtoses[0], conf.chirp_kurtoses[1], size=conf.num_chirps
        )
        chirp_contrasts = np.random.uniform(
            conf.chirp_contrasts[0],
            conf.chirp_contrasts[1],
            size=conf.num_chirps,
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

        rise_times = np.random.uniform(
            0, conf.simulation_duration_rec, size=conf.num_rises
        )
        rise_sizes = np.random.uniform(
            conf.rise_sizes[0], conf.rise_sizes[1], size=conf.num_rises
        )
        rise_rise_taus = np.random.uniform(
            conf.rise_rise_taus[0], conf.rise_rise_taus[1], size=conf.num_rises
        )
        rise_decay_taus = np.random.uniform(
            conf.rise_decay_taus[0],
            conf.rise_decay_taus[1],
            size=conf.num_rises,
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

    for e in range(conf.num_electrodes):
        # draw random values to generate noise
        noise_std = np.random.uniform(conf.noise_stds[0], conf.noise_stds[1])
        noise = noise_std * np.random.randn(len(eod))
        recording += noise
        spec = spectrogram_of(torch.tensor(recording).float().to(device))

        if e == 0:
            spec = spec
        else:
            spec += spec

    spec = in_decibel(spec).cpu().numpy() / conf.num_electrodes

    # get time and frequency axis
    spec_times = np.arange(0, spec.shape[1]) * hop_length / conf.samplerate
    frequencies = np.arange(0, spec.shape[0]) * conf.samplerate / nfft

    traces_cropped, trace_ids, trace_idx = [], [], []
    spec_min, spec_max = np.min(spec_times), np.max(spec_times)

    for fish, trace in enumerate(traces):
        traces_cropped.append(trace[(time >= spec_min) & (time <= spec_max)])
        trace_ids.append(np.ones_like(traces_cropped[-1]) * fish)
        trace_idx.append(np.arange(len(traces_cropped[-1])))

    time_cropped = time[(time >= spec_min) & (time <= spec_max)]

    outpath = pathlib.Path(conf.testing_data_path)
    outpath.mkdir(parents=True, exist_ok=True)

    np.save(outpath / "fill_spec.npy", spec)
    np.save(outpath / "fill_freqs.npy", frequencies)
    np.save(outpath / "fill_times.npy", spec_times)
    np.save(outpath / "fund_v.npy", np.ravel(traces_cropped))
    np.save(outpath / "ident_v.npy", np.ravel(trace_ids))
    np.save(outpath / "idx_v.npy", np.ravel(trace_idx))
    np.save(outpath / "times.npy", time_cropped)
    np.save(outpath / "correct_chirp_times.npy", np.ravel(correct_chirp_times))
    np.save(
        outpath / "correct_chirp_time_ids.npy", np.ravel(correct_chirp_time_ids)
    )


if __name__ == "__main__":
    main()

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

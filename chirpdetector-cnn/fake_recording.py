#!/usr/bin/env python3

import pathlib

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from models.modelhandling import check_device
from simulations.fish_signal import chirps, rises, wavefish_eods
from utils.datahandling import find_on_time
from utils.filehandling import ConfLoader, NumpyLoader
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


def scale(x, mu, std):
    x -= np.mean(x)
    x /= np.std(x)
    x = (x * std) + mu
    return x


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
        conf.chirp_durations[0], conf.chirp_durations[1] * 20, size=num_zeros
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

        # make noise
        noise = np.random.normal(0, std, size=len(recording))

        # cut noise to size
        noise = noise[(time >= start) & (time < end)]

        # add amplitude modulated
        ampmod = np.linspace(0, 1, len(noise))[::-1]

        noise *= ampmod

        recording[(time >= start) & (time < end)] += noise

    return recording


def add_background_noise(recording):
    noise_std = np.random.uniform(
        conf.background_noise_stds[0], conf.background_noise_stds[1]
    )
    noise = np.random.normal(0, noise_std, size=len(recording))
    recording += noise

    return recording


def natural_scale(recording):
    mu = 0
    std = np.random.uniform(
        conf.natural_std_range[0], conf.natural_std_range[1]
    )
    recording = scale(recording, mu, std)

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
        trace = add_noise(trace)

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
            recording = eod
        else:
            recording += eod

    # add regular background noise
    recording = add_background_noise(recording)

    # add these pesky vertical noise bands
    recording = add_vertical_noise_bands(time, recording)

    # scale to match units in natural recordings
    recording = natural_scale(recording)

    # reformat the tracks and chirp times
    traces = np.concatenate(traces)
    trace_ids = np.concatenate(trace_ids)
    correct_chirp_times = np.concatenate(correct_chirp_times)
    correct_chirp_time_ids = np.concatenate(correct_chirp_time_ids)

    # compute and plot spectrogram
    spec, spec_times, spec_freqs = to_spectrogram(recording)

    # crop original data to fit spectrogram time axis
    traces_c, trace_ids_c, trace_idx_c = [], [], []
    start, stop = spec_times[0], spec_times[-1]
    for fish_id in np.unique(trace_ids):
        trace = traces[trace_ids == fish_id][(time >= start) & (time < stop)]
        traces_c.append(trace)
        trace_ids_c.append(np.ones_like(trace) * fish_id)
        trace_idx_c.append(np.arange(len(trace)))
    time_c = time[(time >= start) & (time < stop)]

    track_freqs = np.concatenate(traces_c)
    track_idents = np.concatenate(trace_ids_c)
    track_indices = np.concatenate(trace_idx_c)
    track_times = time_c

    # save data to numpy files
    outpath = pathlib.Path(conf.testing_data_path)
    outpath.mkdir(parents=True, exist_ok=True)
    np.save(outpath / "fill_spec.npy", spec)
    np.save(outpath / "fill_times.npy", spec_times)
    np.save(outpath / "fill_freqs.npy", spec_freqs)
    np.save(outpath / "raw.npy", recording)
    np.save(outpath / "fund_v.npy", track_freqs)
    np.save(outpath / "ident_v.npy", track_idents)
    np.save(outpath / "idx_v.npy", track_indices)
    np.save(outpath / "times.npy", track_times)
    np.save(outpath / "correct_chirp_times.npy", correct_chirp_times)
    np.save(outpath / "correct_chirp_time_ids.npy", correct_chirp_time_ids)

    # TODO: Maybe add the vertical noise bands to blacklisted times?


def main():
    fake_recording()

    d = NumpyLoader(conf.testing_data_path)

    fig, ax = plt.subplots(
        figsize=(24 * ps.cm, 10 * ps.cm), constrained_layout=True
    )

    specshow(
        d.fill_spec,
        d.fill_times,
        d.fill_freqs,
        ax,
        aspect="auto",
        origin="lower",
    )

    for track_id in np.unique(d.ident_v):
        track = d.fund_v[d.ident_v == track_id]
        track_times = d.times[d.idx_v[d.ident_v == track_id]]
        ax.plot(track_times, track, color=ps.black, lw=1)

        id_chirp_times = d.correct_chirp_times[
            d.correct_chirp_time_ids == track_id
        ]
        time_index = np.searchsorted(track_times, id_chirp_times)
        freq_index = [
            find_on_time(track_times, chirp, False) for chirp in id_chirp_times
        ]
        ax.scatter(
            track_times[time_index],
            track[freq_index],
            color=ps.black,
            marker="|",
        )

    ax.set_xlim(d.fill_times[0], d.fill_times[-1])
    ax.set_ylim(np.min(d.fund_v) - 300, np.max(d.fund_v) + 300)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    plt.show()


if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import resample 
from IPython import embed
from thunderfish.powerspectrum import spectrogram, decibel

from utils.logger import make_logger
from utils.filehandling import ConfLoader
from simulations.fish_signal import chirps, rises, wavefish_eods

conf = ConfLoader("config.yml")
logger = make_logger(__name__)

def recording():
    logger.info("Generating fake recording")

    time = np.arange(
            0, 
            conf.simulation_duration_rec, 
            1/conf.samplerate
    )

    eodfs = np.random.randint(
            conf.eodfs[0], 
            conf.eodfs[1], 
            size=conf.num_fish
    )

    traces = []

    for fish, eodf in enumerate(eodfs): 

        chirp_times = np.random.uniform(
                0, time[-1], size=conf.num_chirps
        )
        chirp_sizes = np.random.uniform(
                conf.chirp_sizes[0], 
                conf.chirp_sizes[1], 
                size=conf.num_chirps
        )
        chirp_durations = np.random.uniform(
                conf.chirp_durations[0], 
                conf.chirp_durations[1], 
                size=conf.num_chirps
        )
        chirp_kurtoses = np.random.uniform(
                conf.chirp_kurtoses[0], 
                conf.chirp_kurtoses[1], 
                size=conf.num_chirps
        )
        chirp_contrasts = np.random.uniform(
                conf.chirp_contrasts[0], 
                conf.chirp_contrasts[1], 
                size=conf.num_chirps
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
                conf.rise_sizes[0],
                conf.rise_sizes[1],
                size=conf.num_rises
        )
        rise_rise_taus = np.random.uniform(
                conf.rise_rise_taus[0],
                conf.rise_rise_taus[1],
                size=conf.num_rises
        )
        rise_decay_taus = np.random.uniform(
                conf.rise_decay_taus[0],
                conf.rise_decay_taus[1],
                size=conf.num_rises
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

        eod = wavefish_eods(
                "Alepto", 
                eod_trace, 
                conf.samplerate, 
                conf.simulation_duration_rec, 
                phase0=0, 
                noise_std=0.01
        )

        eod = eod * amplitude_modulation

        if fish == 0:
            recording = eod
        else:
            recording += eod

        

    recording = recording / len(eodfs)

    spec, frequencies, spec_times = spectrogram(
            data = recording, 
            ratetime = conf.samplerate, 
            freq_resolution = conf.frequency_resolution, 
            overlap_frac = conf.overlap_fraction
    )
    spec = decibel(spec)
    plt.imshow(spec, aspect="auto", origin="lower")
    plt.show()

    traces_cropped, trace_ids = [], []
    spec_min, spec_max = np.min(spec_times), np.max(spec_times)
    for fish, trace in enumerate(traces):
        traces_cropped.append(trace[(time >= spec_min) & (time <= spec_max)])
        trace_ids.append(np.ones_like(traces_cropped[-1]) * fish)

    np.save(conf.dataroot_testing + "/spectrogram.npy", spec)
    np.save(conf.dataroot_testing + "/frequencies.npy", frequencies)
    np.save(conf.dataroot_testing + "/times.npy", spec_times)
    np.save(conf.dataroot_testing + "/traces.npy", np.ravel(traces_cropped))
    np.save(conf.dataroot_testing + "/trace_ids.npy", np.ravel(trace_ids))

    return time, recording, traces


if __name__ == "__main__":
    time, recording, traces = recording()

    plt.plot(time, recording)
    plt.show()
    for fish, eod_trace in enumerate(traces):
        plt.plot(time, eod_trace)
    plt.show()

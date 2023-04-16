import math
import pathlib

import matplotlib.pyplot as plt
import nixio as nio
import numpy as np
import torch
from IPython import embed
from thunderfish.dataloader import DataLoader
from torchaudio.transforms import AmplitudeToDB, Spectrogram

from utils.filehandling import ConfLoader, LoadData

conf = ConfLoader("config.yml")


def next_power_of_two(num):
    """
    Takes a float as input and returns the next power of two.

    Args:
        num (float): The input number.

    Returns:
        float: The next power of two.
    """
    # Check if the input is already a power of two
    if math.log2(num).is_integer():
        return num

    # Find the next power of two using log2 and ceil
    next_pow = math.ceil(math.log2(num))

    # Return the result
    return 2**next_pow


def freqres_to_nfft(freq_res, samplingrate):
    """
    Convert the frequency resolution of a spectrogram to
    the number of FFT bins.
    """
    return next_power_of_two(samplingrate / freq_res)


def overlap_to_hoplen(overlap, nfft):
    """
    Convert the overlap of a spectrogram to the hop length.
    """
    return int(np.floor(nfft * (1 - overlap)))


def safe_int(num):
    """
    Convert a float to an int without rounding.
    """
    if num.is_integer():
        return int(num)
    else:
        raise ValueError("Number is not an integer.")


def imshow(spec, time, freq):
    """
    Plot a spectrogram.
    """
    # plt.pcolormesh(time, freq, spec)
    plt.imshow(
        spec,
        aspect="auto",
        origin="lower",
        extent=[np.min(time), np.max(time), np.min(freq), np.max(freq)],
        interpolation="none",
    )
    plt.ylim(0, 2000)
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path = "/home/weygoldt/projects/chirpdetector/data/2022-06-02-10_00/"

    data = LoadData(path)

    buffersize = 20
    samplingrate = data.raw.samplerate
    nelectrodes = data.raw.shape[1]

    nfft = freqres_to_nfft(conf.frequency_resolution, samplingrate)
    hop_length = overlap_to_hoplen(conf.overlap_fraction, nfft)
    chunk_size = samplingrate * buffersize
    padding = 1 * samplingrate  # padding of raw singnal to limit edge effects

    # Good window for this recording
    window_start_index = (3 * 60 * 60 + 6 * 60 + 43) * samplingrate
    window_stop_index = window_start_index + 180 * samplingrate
    signal = data.raw[window_start_index:window_stop_index]
    nchunks = math.ceil(signal.shape[0] / chunk_size)

    # collect frequency traces
    tracks = []
    idents = []
    indices = []

    window_start_secods = window_start_index / samplingrate
    window_stop_seconds = window_stop_index / samplingrate
    track_time = data.time[
        (data.time >= window_start_secods) & (data.time <= window_stop_seconds)
    ]

    for track_id in np.unique(data.ident):
        track = data.freq[data.ident == track_id]
        time = data.time[data.idx[data.ident == track_id]]
        index = data.idx[data.ident == track_id]

        # snip the track to the window
        track = track[
            (time >= window_start_secods) & (time <= window_stop_seconds)
        ]
        index = index[
            (time >= window_start_secods) & (time <= window_stop_seconds)
        ]
        ids = np.full(len(track), track_id)

        tracks.append(track)
        indices.append(index)
        idents.append(ids)

    tracks = np.concatenate(tracks)
    track_indices = np.concatenate(indices)
    track_idents = np.concatenate(idents)
    track_time = data.time - window_start_secods
    track_indices -= len(track_time[track_time < 0])
    track_time = track_time[
        (track_time >= 0)
        & (track_time <= window_stop_seconds - window_start_secods)
    ]

    spectrogram_of = Spectrogram(
        n_fft=nfft,
        hop_length=hop_length,
        power=2,
        normalized=True,
    ).to(device)

    in_decibel = AmplitudeToDB(stype="power", top_db=80).to(device)

    file = nio.File.open("../real_data/dataset.nix", nio.FileMode.Overwrite)
    block = file.create_block("spectrogram", "spectrogram")

    timetracker = 0
    for i in range(nchunks):
        print(f"Chunk {i + 1} of {nchunks}")

        for electrode in range(nelectrodes):
            # get chunk for current electrode
            # add overlap depending if first, middle or last chunk
            if i == 0:
                idx1 = safe_int(i * chunk_size)
                idx2 = safe_int((i + 1) * chunk_size + padding)
                chunk = signal[idx1:idx2, electrode]
            elif i == nchunks - 1:
                idx1 = safe_int(i * chunk_size - padding)
                idx2 = safe_int((i + 1) * chunk_size)
                chunk = signal[idx1:idx2, electrode]
            else:
                idx1 = safe_int((i * chunk_size - padding))
                idx2 = safe_int((i + 1) * chunk_size + padding)
                chunk = signal[idx1:idx2, electrode]

            # compute how much padding to remove from the start and end of the spec
            # to get the correct time axis
            spec_padding = int(padding // hop_length)

            # convert to tensor and into gpu
            chunk = torch.from_numpy(chunk).to(device)

            # calculate spectrogram
            chunk_spec = spectrogram_of(chunk)

            # remove padding from spectrogram
            if i == 0:
                chunk_spec = chunk_spec[:, :-spec_padding]
            elif i == nchunks - 1:
                chunk_spec = chunk_spec[:, spec_padding:]
            else:
                chunk_spec = chunk_spec[:, spec_padding:-spec_padding]

            # convert to decibel
            chunk_spec = in_decibel(chunk_spec)

            # sum up the spectrograms
            if electrode == 0:
                spec = chunk_spec
            else:
                spec += chunk_spec

        # get chunk for all electrodes for saving
        idx1 = safe_int(i * chunk_size)
        idx2 = safe_int((i + 1) * chunk_size)
        chunk = signal[idx1:idx2, :]

        # normalize by number of electrodes
        spec = spec / nelectrodes

        # convert to numpy and into ram
        spec = spec.cpu().numpy()

        # get time and frequency axis
        time = (
            np.arange(0, spec.shape[1]) * hop_length / samplingrate
            + timetracker
        )
        freq = np.arange(0, spec.shape[0]) * samplingrate / nfft

        # keep track of the time for the next iteration
        timetracker = time[-1]

        # create the data arrays on disk in the first iteration
        if i == 0:
            save_raw = block.create_data_array(
                "raw", "raw signal", data=chunk, unit="mV"
            )
            save_raw.append_sampled_dimension(
                sampling_interval=1 / samplingrate, label="time", unit="s"
            )
            save_spec = block.create_data_array(
                "spec", "spectrogram matrix", data=spec, unit="dB"
            )
            save_time = block.create_data_array(
                "spec_time", "time axis", data=time, unit="s"
            )
            save_freq = block.create_data_array(
                "spec_freq", "frequency axis", data=freq, unit="Hz"
            )
        else:
            # frequency axis never changes so we can skip it in the following
            save_raw.append(chunk, axis=0)
            save_spec.append(spec, axis=1)
            save_time.append(time, axis=0)

    # also add the other data to the nix file
    save_tracks = block.create_data_array(
        "track_freqs", "frequency tracks", data=tracks, unit="Hz"
    )
    save_idents = block.create_data_array(
        "track_idents", "track identities", data=track_idents
    )
    save_indices = block.create_data_array(
        "track_indices", "track indices", data=track_indices
    )
    save_time = block.create_data_array(
        "track_times", "time axis", data=track_time, unit="s"
    )
    file.close()


if __name__ == "__main__":
    main()

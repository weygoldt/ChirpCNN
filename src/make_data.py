#!/usr/bin/env python3

import os
import uuid
from itertools import product
import shutil

import cv2
from tqdm import tqdm
from IPython import embed
from thunderfish.powerspectrum import spectrogram, decibel
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse 

from simulations.fish_signal import chirps, wavefish_eods
from utils.datahandling import resize_image

# chirp_path = "../data/chirp/"
# nochirp_path = "../data/nochirp/"

# define parameters for the waveform simulation 
samplerate = 20000
simulation_duration = 2

# define parameters for the spectrogram
freq_resolution = 5
overlap_frac = 0.99

# define chirp parameter boundaries
chirp_times = [1]
eodf = (300, 1500)
size = (90, 200)
duration = (0.02, 0.09)
kurtosis = (0.8, 1.2)
contrast = (0.01, 0.05)

# define how many levels of each parameter to test
levels = 5 # CAREFUL! This increases the dataset size by a factor of nparams^5!

# define spectrogram ROI padding around chirp
time_pad = 0.12 # seconds before and after chirp, symetric
freq_pad = (100, 300) # freq above and below chirp, unsymetric
time_center_jitter = 0.01 # seconds to offset the center of the ROI

# define transformation params before saving 
imgsize = 128

def make_chirps(path, debug = False):

    assert path[-1] == "/", "Path must end with a slash"
    assert debug in [True, False], "Debug must be True or False"

    # make the chirp parameter arrays
    eodfs = np.linspace(eodf[0], eodf[1], levels).tolist() 
    sizes = np.linspace(size[0], size[1], levels).tolist()
    durations = np.linspace(duration[0], duration[1], levels).tolist()
    kurtosiss = np.linspace(kurtosis[0], kurtosis[1], levels).tolist()
    contrasts = np.linspace(contrast[0], contrast[1], levels).tolist()

    # make all possible combinations of chirp parameters
    all_params = [eodfs, sizes, durations, kurtosiss, contrasts]
    all_params = np.asarray(list(product(*all_params)))

    # jitter the center of the ROI
    jitters = np.random.uniform(
            -time_center_jitter,
            time_center_jitter, 
            size = len(all_params)
    ).tolist()

    # pick some random ones to plot
    if debug:
        subset_index = np.random.randint(0, len(all_params) + 1, size=20)
        subset = all_params[subset_index]
    else:
        subset = all_params

    total = len(subset)
    for iter, params in tqdm(enumerate(subset), total = total, desc = "Making chirps"):

        ones = np.ones_like(chirp_times)

        params = {
            'eodf': params[0],
            'chirp_size': params[1] * ones,
            'chirp_width': params[2] * ones,
            'chirp_kurtosis': params[3] * ones,
            'chirp_contrast': params[4] * ones,
        }

        chirp_trace, ampmod = chirps(
                samplerate = samplerate, 
                duration = simulation_duration, 
                chirp_times = chirp_times, 
                **params
        )

        signal = wavefish_eods(
                fish="Alepto", 
                frequency=chirp_trace, 
                samplerate=samplerate, 
                duration=simulation_duration, 
                phase0=0.0, 
                noise_std=0.01
        )

        signal = signal * ampmod
        time = np.arange(0, simulation_duration, 1/samplerate)

        spec, freqs, spectime = spectrogram(
            data=signal,
            ratetime=samplerate,
            freq_resolution=freq_resolution,
            overlap_frac=overlap_frac,
        )

        fullspec = decibel(spec)

        # define the region of interest
        time_center = chirp_times[0] + jitters[iter]
        xroi = (time_center-time_pad, time_center+time_pad)
        yroi = (params['eodf'] - freq_pad[0], params['eodf'] + freq_pad[1])

        # crop spec to the region of interest
        spec = fullspec[(freqs > yroi[0]) & (freqs < yroi[1]), :]
        spec = spec[:, (spectime > xroi[0]) & (spectime < xroi[1])]
    
        # normalize the spectrogram between 0 and 1
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

        # pad the spectrogram symetrically to 128x128
        spec = resize_image(spec, imgsize)

        # save the chirps
        np.save(path + str(uuid.uuid1()), spec)

        if not debug:
            continue

        ylims = (params['eodf'] - 150, params['eodf'] + 350)
        fig, axs = plt.subplots(3, 1, sharex = True)
        axs[0].plot(time, chirp_trace)
        axs[1].plot(time, signal)
        axs[2].imshow(
                fullspec, 
                aspect = 'auto', 
                origin = 'lower', 
                extent = [spectime[0], spectime[-1], freqs[0], freqs[-1]], 
                interpolation = 'none', 
        )
        rect = patches.Rectangle(
                (xroi[0], yroi[0]), 
                xroi[1]-xroi[0], 
                yroi[1]-yroi[0], 
                linewidth=1, 
                edgecolor='r', 
                facecolor='none'
        )
        axs[2].add_patch(rect)
        axs[2].set_ylim(*ylims)
        plt.show()

        plt.imshow(
                spec, 
                origin = 'lower', 
                interpolation = 'none', 
                cmap = 'gray'
        )
        plt.show()

        print(np.shape(spec))
        print(np.min(spec), np.max(spec))

    return total 


def make_shifted_chirps(path, debug = False):

    assert path[-1] == "/", "Path must end with a slash"
    assert debug in [True, False], "Debug must be True or False"

    # make the chirp parameter arrays
    eodfs = np.linspace(eodf[0], eodf[1], levels).tolist() 
    sizes = np.linspace(size[0], size[1], levels).tolist()
    durations = np.linspace(duration[0], duration[1], levels).tolist()
    kurtosiss = np.linspace(kurtosis[0], kurtosis[1], levels).tolist()
    contrasts = np.linspace(contrast[0], contrast[1], levels).tolist()

    # make all possible combinations of chirp parameters
    all_params = [eodfs, sizes, durations, kurtosiss, contrasts]
    all_params = np.asarray(list(product(*all_params)))

    # jitter the center of the ROI
    jitters1 = np.random.uniform(
            - 0.3,
            - time_center_jitter - 0.1, 
            size = int(len(all_params)/2), 
    ) 
    jitters2 = np.random.uniform(
            time_center_jitter + 0.1, 
            0.3,
            size = len(all_params) - len(jitters1)
    )
    jitters = np.random.permutation(np.append(jitters1, jitters2))

    # pick some random ones to plot
    if debug:
        subset_index = np.random.randint(0, len(all_params) + 1, size=20)
        subset = all_params[subset_index]
    else:
        subset = all_params

    total = len(subset)
    for iter, params in tqdm(enumerate(subset), total = total, desc = "Making chirps"):

        ones = np.ones_like(chirp_times)

        params = {
            'eodf': params[0],
            'chirp_size': params[1] * ones,
            'chirp_width': params[2] * ones,
            'chirp_kurtosis': params[3] * ones,
            'chirp_contrast': params[4] * ones,
        }

        chirp_trace, ampmod = chirps(
                samplerate = samplerate, 
                duration = simulation_duration, 
                chirp_times = chirp_times, 
                **params
        )

        signal = wavefish_eods(
                fish="Alepto", 
                frequency=chirp_trace, 
                samplerate=samplerate, 
                duration=simulation_duration, 
                phase0=0.0, 
                noise_std=0.01
        )

        signal = signal * ampmod
        time = np.arange(0, simulation_duration, 1/samplerate)

        spec, freqs, spectime = spectrogram(
            data=signal,
            ratetime=samplerate,
            freq_resolution=freq_resolution,
            overlap_frac=overlap_frac,
        )

        fullspec = decibel(spec)

        # define the region of interest
        time_center = chirp_times[0] + jitters[iter]
        xroi = (time_center-time_pad, time_center+time_pad)
        yroi = (params['eodf'] - freq_pad[0], params['eodf'] + freq_pad[1])

        # crop spec to the region of interest
        spec = fullspec[(freqs > yroi[0]) & (freqs < yroi[1]), :]
        spec = spec[:, (spectime > xroi[0]) & (spectime < xroi[1])]
    
        # normalize the spectrogram between 0 and 1
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

        # pad the spectrogram symetrically to 128x128
        spec = resize_image(spec, imgsize)

        # save the chirps
        np.save(path + str(uuid.uuid1()), spec)

        if not debug:
            continue

        ylims = (params['eodf'] - 150, params['eodf'] + 350)
        fig, axs = plt.subplots(3, 1, sharex = True)
        axs[0].plot(time, chirp_trace)
        axs[1].plot(time, signal)
        axs[2].imshow(
                fullspec, 
                aspect = 'auto', 
                origin = 'lower', 
                extent = [spectime[0], spectime[-1], freqs[0], freqs[-1]], 
                interpolation = 'none', 
        )
        rect = patches.Rectangle(
                (xroi[0], yroi[0]), 
                xroi[1]-xroi[0], 
                yroi[1]-yroi[0], 
                linewidth=1, 
                edgecolor='r', 
                facecolor='none'
        )
        axs[2].add_patch(rect)
        axs[2].set_ylim(*ylims)
        plt.show()

        plt.imshow(
                spec, 
                origin = 'lower', 
                interpolation = 'none', 
                cmap = 'gray'
        )
        plt.show()

        print(np.shape(spec))
        print(np.min(spec), np.max(spec))

    return total 


def make_nochirps(path, dataset_size, debug = False):
    
    assert path[-1] == "/", "Path must end with a slash"
    assert debug in [True, False], "Debug must be True or False"

    # make array of eodfs 
    eodfs = np.linspace(eodf[0], eodf[1], dataset_size).tolist()

    for iter, f in tqdm(enumerate(eodfs), total = dataset_size, desc = "Making nochirps"):

        freq_trace = np.ones(simulation_duration * samplerate) * f

        signal = wavefish_eods(
                fish="Alepto", 
                frequency=freq_trace, 
                samplerate=samplerate, 
                duration=simulation_duration, 
                phase0=0.0, 
                noise_std=0.05
        )

        time = np.arange(0, simulation_duration, 1/samplerate)

        spec, freqs, spectime = spectrogram(
            data=signal,
            ratetime=samplerate,
            freq_resolution=freq_resolution,
            overlap_frac=overlap_frac,
        )

        fullspec = decibel(spec)

        # define the region of interest
        xroi = (chirp_times[0]-time_pad, chirp_times[0]+time_pad)
        yroi = (f - freq_pad[0], f + freq_pad[1])

        # crop spec to the region of interest
        spec = fullspec[(freqs > yroi[0]) & (freqs < yroi[1]), :]
        spec = spec[:, (spectime > xroi[0]) & (spectime < xroi[1])]

        # normalize the spectrogram
        spec = (spec - np.min(spec)) / (np.max(spec) - np.min(spec))

        # resize the spectrogram
        spec = resize_image(spec, imgsize)

        # save the nochirp traces
        np.save(path + str(uuid.uuid1()), spec)

        if not debug:
            continue

        fig, axs = plt.subplots(3, 1, sharex = True)
        axs[0].plot(time, freq_trace)
        axs[1].plot(time, signal)
        axs[2].imshow(
                fullspec,
                aspect = 'auto',
                origin = 'lower',
                extent = [spectime[0], spectime[-1], freqs[0], freqs[-1]],
                interpolation = 'none',
        )
        plt.show()

        plt.imshow(
                spec,
                origin = 'lower',
                interpolation = 'none',
                cmap = 'gray'
        )
        plt.show()


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="../data/")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wipe", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = interface()
    chirp_path = args.path + "chirp/"
    nochirp_path = args.path + "nochirp/"

    if args.wipe & os.path.exists(chirp_path):
        shutil.rmtree(chirp_path)
    if args.wipe & os.path.exists(nochirp_path):
        shutil.rmtree(nochirp_path)

    if os.path.exists(chirp_path) == False:
        os.mkdir(chirp_path)
    if os.path.exists(nochirp_path) == False:
        os.mkdir(nochirp_path)
        
    dataset_size = make_chirps(chirp_path, debug = args.debug)
    make_shifted_chirps(nochirp_path, debug = args.debug)
    # make_nochirps(nochirp_path, dataset_size, debug = args.debug)

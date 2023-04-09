#!/usr/bin/env python3

import shutil
import uuid
import pathlib 

import numpy as np
import argparse 
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.patches import Rectangle
from IPython import embed

from utils.logger import make_logger
from utils.filehandling import ConfLoader, NumpyLoader
from utils.datahandling import find_on_time, resize_image
from utils.plotstyle import PlotStyle
from models.modelhandling import load_model

logger = make_logger(__name__) 
conf = ConfLoader("config.yml")
ps = PlotStyle()

class ChirpExtractor:
    def __init__(self, dataset):
        logger.info("Initializing chirp extractor...")

        self.data = dataset 
        self.samplerate = conf.samplerate
        self.fill_samplerate = 1/np.mean(np.diff(self.data.fill_times))
        self.freq_pad = conf.freq_pad
        self.time_pad = conf.time_pad
        self.window_size = int(conf.time_pad * 2 * self.fill_samplerate)
        self.stride = int(conf.stride * self.fill_samplerate)

        if (self.data.times[-1] // 600 != 0) and (self.mode == "memory"):
            logger.warning("It is recommended to process recordings longer than 10 minutes using the 'disk' mode")

        if self.window_size % 2 == 0:
            self.window_size += 1
            logger.info(f"Time padding is not odd. Adding one.")

        if self.stride % 2 == 0:
            self.stride += 1
            logger.info(f"Stride is not odd. Adding one.")

    def extract(self):

        logger.info("Extracting chirps...")

        first_index = 0
        last_index = self.data.fill_times.shape[0]
        window_start_indices = np.arange(
                first_index, last_index - self.window_size, self.stride, dtype=int
                )

        for track_id in np.unique(self.data.ident_v):
            logger.info(f"Processing track {track_id}...")
            track = self.data.fund_v[self.data.ident_v == track_id]

            chirp_times = self.data.correct_chirp_times[
                    self.data.correct_chirp_time_ids == track_id
                    ]
            snippets = []
            center_t = []

            for window_start_index in window_start_indices:
                
                # Make index were current window will end
                window_end_index = window_start_index + self.window_size

                # Get the current frequency from the track
                center_idx = int(window_start_index + np.floor(self.window_size / 2) + 1)
                window_center_t = self.data.fill_times[center_idx]
                track_index = find_on_time(self.data.times, window_center_t)
                center_freq = track[track_index]

                # From the track frequency compute the frequency
                # boundaries 

                freq_min = center_freq + self.freq_pad[0]
                freq_max = center_freq + self.freq_pad[1]
                
                # Find these values on the frequency axis of the spectrogram
                freq_min_index = find_on_time(self.data.fill_freqs, freq_min)
                freq_max_index = find_on_time(self.data.fill_freqs, freq_max)

                # Using window start, stop and freq lims, extract snippet from spec
                snippet = self.data.fill_spec[
                        freq_min_index:freq_max_index,
                        window_start_index:window_end_index,
                ]

                # Normalize snippet
                snippet = (snippet - np.min(snippet)) / (np.max(snippet) - np.min(snippet))

                # Resize snippet
                snippet = resize_image(snippet, conf.img_size_px)

                # Append snippet to list
                snippets.append(snippet)
                center_t.append(window_center_t)
                
            center_t = np.asarray(center_t)
            chirp_times = np.asarray(sorted(chirp_times))
            spec_chirp_idx = np.asarray([find_on_time(center_t, t, limit=False) for t in chirp_times])
            spec_chirp_times = self.data.fill_times[spec_chirp_idx]
            snippets = np.asarray(snippets).astype(np.float32)

            # Save the chirps
            
            chirppath = pathlib.Path(f"{conf.training_data_path}/chirp")
            chirppath.mkdir(parents=True, exist_ok=True)

            for snip in snippets[spec_chirp_idx]:
                np.save(chirppath / str(uuid.uuid1()), snip)
            logger.info(f"Saved {len(spec_chirp_idx)} chirps")

            # Remove the chirps from the snippets
            snippets = np.delete(snippets, spec_chirp_idx, axis=0)

            nochirppath = pathlib.Path(f"{conf.training_data_path}/nochirp")
            nochirppath.mkdir(parents=True, exist_ok=True)

            for snip in snippets[np.random.choice(len(snippets), len(spec_chirp_idx))]:
                np.save(nochirppath / str(uuid.uuid1()), snip)
            logger.info(f"Saved {len(spec_chirp_idx)} non chirps")


def main():
    d = NumpyLoader(conf.testing_data_path)
    ext = ChirpExtractor(d)
    ext.extract()

if __name__ == "__main__":
    main()

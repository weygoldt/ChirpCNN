#!/usr/bin/env python3

import shutil

import numpy as np
import argparse 
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from matplotlib.patches import Rectangle
from pathlib import Path
from IPython import embed

from utils.logger import make_logger
from utils.filehandling import ConfLoader, NumpyLoader
from utils.datahandling import find_on_time, resize_image
from utils.plotstyle import PlotStyle
from models.modelhandling import load_model

logger = make_logger(__name__) 
conf = ConfLoader("config.yml")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ps = PlotStyle()

class Detector:
    def __init__(self, modelpath, dataset, mode):
        assert mode in ["memory", "disk"], "Mode must be either 'memory' or 'disk'"
        logger.info("Initializing detector...")

        self.mode = mode
        self.model = load_model(modelpath)
        self.data = dataset 
        self.samplerate = conf.samplerate
        self.fill_samplerate = 1/np.mean(np.diff(self.data.fill_times))
        self.freq_pad = conf.freq_pad
        self.time_pad = conf.time_pad
        self.window_size = int(conf.time_pad * 2 * self.fill_samplerate)
        self.stride = int(conf.stride * self.fill_samplerate)

        if (self.data.times[-1] // 600 != 0) and (self.mode == "memory"):
            logger.warning("It is recommended to process recordings longer than 10 minutes using the 'disk' mode")

        if self.window_size % 2 != 0:
            self.window_size += 1
            logger.info(f"Time padding is not even. Please change to an even number.")

        if self.stride % 2 != 0:
            self.stride += 1
            logger.info(f"Stride is not even. Please change to an even number.")

    def detect(self):
        logger.info("Detecting...")

        if self.mode == "memory":
            self._detect_memory()
        else:
            self._detect_disk()

        logger.info(f"Detection complete! Results saved to {conf.detection_data_path}")

    def _detect_memory(self):

        logger.info("Processing in memory...")

        first_index = 0
        last_index = self.data.fill_times.shape[0]
        window_start_indices = np.arange(
                first_index, last_index - self.window_size, self.stride, dtype=int
                )

        for track_id in np.unique(self.data.ident_v):
            logger.info(f"Processing track {track_id}...")
            track = self.data.fund_v[self.data.ident_v == track_id]

            snippets = []

            for window_start_index in window_start_indices:
                
                # Make index were current window will end
                window_end_index = window_start_index + self.window_size

                # Get the current frequency from the track
                window_center_t = self.data.fill_times[window_start_index + self.window_size // 2]
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
                snipped = (snippet - np.min(snippet)) / (np.max(snippet) - np.min(snippet))

                # Resize snippet
                snippet = resize_image(snippet, conf.img_size_px)

                # Add dimension for single channel
                snippet = np.expand_dims(snippet, axis=0)

                # Append snippet to list
                snippets.append(snippet)
                
                """
                fig, ax = plt.subplots()
                ax.imshow(
                        self.data.fill_spec, 
                        aspect="auto",
                        origin="lower",
                        extent=[
                            self.data.fill_times[0],
                            self.data.fill_times[-1],
                            self.data.fill_freqs[0],
                            self.data.fill_freqs[-1],
                        ],
                        cmap="magma",
                )
                # Create a Rectangle patch
                rect = Rectangle(
                        (self.data.fill_times[window_start_index], self.data.fill_freqs[freq_min_index]),
                        self.data.fill_times[window_end_index] - self.data.fill_times[window_start_index],
                        self.data.fill_freqs[freq_max_index] - self.data.fill_freqs[freq_min_index],
                        linewidth=1, 
                        facecolor='none',
                        edgecolor='white',
                )

                # Add the patch to the Axes
                ax.add_patch(rect)

                # Plot the track
                ax.plot(self.data.times, track, linewidth=1)

                # Plot the window center 
                ax.plot(
                        [self.data.fill_times[window_start_index + self.window_size // 2]],
                        [center_freq],
                        marker="o",
                )
                plt.show()
                """

            # Convert snippets to tensor and create dataloader
            snippets = np.asarray(snippets).astype(np.float32)
            print(len(snippets))
        
            for snip in snippets[np.random.randint(0, len(snippets), 10)]:
                plt.imshow(snip[0], cmap="magma", origin="lower")
                plt.show()

            snippets_tensor = torch.from_numpy(snippets)

            with torch.no_grad():
                outputs = self.model(snippets_tensor)

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(probs, dim=1)
            predicted_labels = preds.numpy()
            print(len(predicted_labels))

            if len(np.unique(predicted_labels)) > 1:
                logger.info(f"Found {np.sum(predicted_labels == 0)} chirps")

    def _detect_disk(self):
        logger.info("Processing on disk...")

        data_path = Path(conf.detection_data_path + "/detector")
        if data_path.exists():
            logger.info("Removing data from previous run...")
            shutil.rmtree(data_path)
        else: 
            logger.info("Creating directory for detector data...")
            data_path.mkdir(parents=True, exist_ok=True)
        pass
        ...
        
        if conf.disk_cleanup:
            logger.info("Cleaning up detector data...")
            shutil.rmtree(data_path)


def interface():
    parser = argparse.ArgumentParser(description="Detects chirps on spectrograms.")
    parser.add_argument("--path", type=str, default=conf.testing_data_path, help="Path to the dataset to use for detection")
    parser.add_argument("--mode", type=str, default="memory", help="Mode to use for detection. Can be either 'memory' or 'disk'. Defaults to 'memory'.")
    args = parser.parse_args()
    return args

def main():
    args = interface()
    d = NumpyLoader(args.path)
    modelpath = conf.save_dir
    det = Detector(modelpath, d, args.mode)
    det.detect()

if __name__ == "__main__":
    main()

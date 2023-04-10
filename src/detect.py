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
        self.detected_chirps = None
        self.detected_chirp_ids = None

        if (self.data.times[-1] // 600 != 0) and (self.mode == "memory"):
            logger.warning("It is recommended to process recordings longer than 10 minutes using the 'disk' mode")

        if self.window_size % 2 == 0:
            self.window_size += 1
            logger.info(f"Time padding is not odd. Adding one.")

        if self.stride % 2 == 0:
            self.stride += 1
            logger.info(f"Stride is not odd. Adding one.")

    def detect(self):
        logger.info("Detecting...")

        if self.mode == "memory":
            self._detect_memory()
        else:
            self._detect_disk()

    def _detect_memory(self):

        logger.info("Processing in memory...")

        first_index = 0
        last_index = self.data.fill_times.shape[0]
        window_start_indices = np.arange(
                first_index, last_index - self.window_size, self.stride, dtype=int
                )

        detected_chirps = []
        detected_chirp_ids = []

        for track_id in np.unique(self.data.ident_v):
            logger.info(f"Processing track {track_id}...")
            track = self.data.fund_v[self.data.ident_v == track_id]

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

                # Add dimension for single channel
                snippet = np.expand_dims(snippet, axis=0)

                # Append snippet to list
                snippets.append(snippet)
                center_t.append(window_center_t)
                
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
            center_t = np.asarray(center_t)
            
            snippets_tensor = torch.from_numpy(snippets)

            with torch.no_grad():
                outputs = self.model(snippets_tensor)

            _, preds = torch.max(outputs, dim=1)
            predicted_labels = preds.numpy()
            # print(len(predicted_labels))

            if len(np.unique(predicted_labels)) > 1:
                logger.info(f"Found {np.sum(predicted_labels == 0)} chirps")
            
            detected_chirps.append(center_t[predicted_labels == 0])
            detected_chirp_ids.append(np.repeat(track_id, np.sum(predicted_labels == 0)))

        self.detected_chirps = np.concatenate(detected_chirps)
        self.detected_chirp_ids = np.concatenate(detected_chirp_ids)
            

    def _detect_disk(self):
        logger.info("This function is not yet implemented. Aborting ...")
        exit()

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

    def plot(self):

        d = self.data # <----- Quick fix, remove this!!!
        correct_chirps = np.load(conf.testing_data_path + "/correct_chirp_times.npy")
        correct_chirp_ids = np.load(conf.testing_data_path + "/correct_chirp_time_ids.npy")

        fig, ax = plt.subplots(figsize=(24 * ps.cm, 12 * ps.cm), constrained_layout=True)
        ax.imshow(
                d.fill_spec,
                aspect="auto",
                origin="lower",
                extent=[
                    d.fill_times[0],
                    d.fill_times[-1],
                    d.fill_freqs[0],
                    d.fill_freqs[-1],
                ],
                zorder = -20,
        )

        for track_id in np.unique(d.ident_v):

            track_id = int(track_id)
            track = d.fund_v[d.ident_v == track_id]
            freq = np.median(track)

            correct_t = correct_chirps[correct_chirp_ids == track_id]
            findex = np.asarray([find_on_time(d.times, t) for t in correct_t])
            correct_f = track[findex]

            detect_t = self.detected_chirps[self.detected_chirp_ids == track_id]
            findex = np.asarray([find_on_time(d.times, t) for t in detect_t])
            detect_f = track[findex]
            
            ax.plot(d.times, track, linewidth=1, zorder = -10, color=ps.black)
            ax.scatter(correct_t, correct_f, s=20, marker="o", color=ps.black, zorder = 0)
            ax.scatter(detect_t, detect_f, s=10, marker="o", color=ps.white, edgecolor=ps.black, zorder = 10)

        # proxy_artist1 = plt.Line2D([0], [0], color=ps.blue, lw=1, label='EODf tracks')
        # proxy_artist2 = plt.Line2D([0], [0], color=ps.black, ls="none", marker='o', label='Correct chirp positions')
        # proxy_artist3 = plt.Line2D([0], [0], color=ps.gblue3, ls="none", marker='.', label='Detected chirp positions')

        ax.set_ylim(np.min(d.fund_v - 50), np.max(d.fund_v + 150))
        ax.set_xlim(np.min(d.fill_times), np.max(d.fill_times))
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Frequency [Hz]")
        # ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left",
        #             mode="expand", borderaxespad=0, ncol=3,
        #           handles=[proxy_artist1, proxy_artist2, proxy_artist3])

        plt.savefig("../assets/detection.png")
        plt.show()


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
    det.plot()


if __name__ == "__main__":
    main()

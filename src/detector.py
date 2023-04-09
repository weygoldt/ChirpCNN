import shutil

import numpy as np
import argparse 
from pathlib import Path

from utils.logger import make_logger
from utils.filehandling import ConfLoader, NumpyLoader
from models.modelhandling import load_model

logger = make_logger(__name__) 
conf = ConfLoader("config.yml")

class Detector:
    def __init__(self, modelpath, dataset, mode):
        assert mode in ["memory", "disk"], "Mode must be either 'memory' or 'disk'"
        logger.info("Initializing detector...")

        self.mode = mode
        self.model = load_model(modelpath)
        self.data = dataset 
        self.freq_pad = conf.freq_pad
        self.time_pad = conf.time_pad

        if (self.data.times[-1] // 600 != 0) and (self.mode == "memory"):
            logger.warning("It is recommended to process recordings longer than 10 minutes using the 'disk' mode")

    def detect(self):
        logger.info("Detecting...")

        if self.mode == "memory":
            self._detect_memory()
        else:
            self._detect_disk()
        pass 
        ...
        logger.info(f"Detection complete! Results saved to {conf.detection_data_path}")

    def _detect_memory(self):
        logger.info("Processing in memory...")

        for track_id in np.unique(self.data.ident_v):
            logger.info(f"Processing track {track_id}...")
            track = self.data.fund_v[self.data.ident_v == track_id]

            start_time = self.data.times[0] + self.time_pad
            stop_time = self.data.times[-1] - self.time_pad
            window_centers = np.arange()
        ...

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

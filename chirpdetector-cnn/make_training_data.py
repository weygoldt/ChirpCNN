#!/usr/bin/env python3

import detect_chirps as detect_chirps
import extract_training_data
import fake_recording as fake_recording
import train_model as train_model
from utils.filehandling import ConfLoader
from utils.logger import make_logger

conf = ConfLoader("config.yml")
logger = make_logger(__name__)


def main():
    for i in range(conf.generations):
        logger.info(f"Generation {i}/{conf.generations}")
        fake_recording.main("default")
        extract_training_data.main()

    # logger.info("Training model...")
    # train_model.main()

    # logger.info("Creating new dataset ...")
    # fake_recording.main("test")

    # logger.info("Detecting chirps ...")
    # detect_chirps.main()


if __name__ == "__main__":
    main()

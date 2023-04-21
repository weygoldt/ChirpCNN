#!/usr/bin/env python3

import clean
import detect
import make_fake_recording
import train_model
import training_data_from_recording
from utils.filehandling import ConfLoader
from utils.logger import make_logger

conf = ConfLoader("config.yml")
logger = make_logger(__name__)


def main():
    clean.main()

    for i in range(conf.generations):
        logger.info(f"Generation {i}/{conf.generations}")
        make_fake_recording.main("default")
        training_data_from_recording.main()

    logger.info("Training model...")
    train_model.main()

    # logger.info("Creating new dataset ...")
    # make_fake_recording.main()
    # detect.main()


if __name__ == "__main__":
    main()

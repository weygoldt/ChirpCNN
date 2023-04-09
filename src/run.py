#!/usr/bin/env python3

from utils.filehandling import ConfLoader
from utils.logger import make_logger
import make_fake_recording
import training_data_from_recording 
import train_model
import detect
import clean

conf = ConfLoader("config.yml")
logger = make_logger(__name__)

def main():

    clean.main()

    for i in range(conf.generations):
        logger.info(f"Generation {i}/{conf.generations}")
        make_fake_recording.main()
        training_data_from_recording.main()

    logger.info("Training model...")
    train_model.main()

    logger.info("Creating new dataset ...")
    make_fake_recording.main()
    detect.main()


if __name__ == "__main__":
    main()

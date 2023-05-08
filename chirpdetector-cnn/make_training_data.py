#!/usr/bin/env python3

"""
Use the fake_recording module to generate a full training dataset entirely
consisting of fake recordings. Since hybrid simulations are
implemented, this is not particularly useful anymore.
"""

import detect_chirps as detect_chirps
import extract_training_data
import train_model as train_model
from fake_recording import fake_recording
from utils.filehandling import ConfLoader
from utils.logger import make_logger

conf = ConfLoader("config.yml")
logger = make_logger(__name__)


def main():
    for i in range(conf.generations):
        logger.info(f"Generation {i}/{conf.generations}")
        fake_recording()
        extract_training_data.main()


if __name__ == "__main__":
    main()

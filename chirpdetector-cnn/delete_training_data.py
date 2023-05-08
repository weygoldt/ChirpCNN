#!/usr/bin/env python3

"""
Deletes all data in the training data directory.
"""

import pathlib
import shutil

from utils.filehandling import ConfLoader
from utils.logger import make_logger

logger = make_logger(__name__)
conf = ConfLoader("config.yml")


def main():
    logger.info("Cleaning up...")
    if pathlib.Path(conf.training_data_path).exists():
        shutil.rmtree(conf.training_data_path)

    if pathlib.Path(conf.detection_data_path).exists():
        shutil.rmtree(conf.detection_data_path)

    if pathlib.Path(conf.save_dir).exists():
        pathlib.Path(conf.save_dir).unlink(missing_ok=True)

    logger.info("Done.")


if __name__ == "__main__":
    main()

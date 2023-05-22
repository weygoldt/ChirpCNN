#!/usr/bin/env python3

"""
Extracts a snippet from a real grid recording and saves it to disk to 
test the detector on it.
"""

import argparse
import pathlib

from utils.filehandling import ChirpDataset, ChirpDataSubset, Config


def extract_snippet(input_path, output_path, start_t, stop_t):
    cd = ChirpDataset(input_path)
    cs = ChirpDataSubset(cd, start_t, stop_t, on="time")
    cs.save(output_path)


def interface():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        "-ip",
        type=pathlib.Path,
        help="Path to the directory containing the data.",
    )
    parser.add_argument(
        "--output_path",
        "-op",
        type=pathlib.Path,
        help="Path to the directory where the snippet should be saved.",
    )
    parser.add_argument(
        "--start",
        type=float,
        help="Start time of the snippet in seconds.",
    )
    parser.add_argument(
        "--stop",
        type=float,
        help="Stop time of the snippet in seconds.",
    )
    args = parser.parse_args()

    return args


def main():
    args = interface()
    extract_snippet(args.input_path, args.output_path, args.start, args.stop)


if __name__ == "__main__":
    main()

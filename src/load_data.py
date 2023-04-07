import os
import numpy as np
import tensorflow as tf
from IPython import embed

from utils.filehandling import get_files

def load_numpy(file_path):
    file = np.load(file_path)
    file = tf.convert_to_tensor(file, dtype=tf.float32)
    return file

dataroot = "../data"
filepaths, labels, level_dict = get_files(dataroot, ext="*.npy")

dataset = tf.data.Dataset.list_files(filepaths, shuffle=False)
dataset = dataset.map(load_numpy)

embed()
exit()
    

    

# # Create a list to store the file paths
# file_paths = []

# # Loop through the chirp folder and append file paths to the list
# for file_name in os.listdir(chirp_folder):
#     file_path = os.path.join(chirp_folder, file_name)
#     file_paths.append(file_path)

# # Loop through the nochirp folder and append file paths to the list
# for file_name in os.listdir(nochirp_folder):
#     file_path = os.path.join(nochirp_folder, file_name)
#     file_paths.append(file_path)

# def load_npy_file(file_path):
#     spectrogram = np.load(file_path)
#     spectrogram = tf.convert_to_tensor(spectrogram, dtype=tf.float32)
#     return spectrogram

# dataset = tf.data.Dataset.list_files("../data/chirps/*.npy", shuffle=False)

# dataset = dataset.map(load_npy_file)




import os
import numpy as np
from IPython import embed   
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

# Define a custom dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.class_labels = os.listdir(root_dir)
        self.file_paths = []
        self.labels = []

        for i, class_label in enumerate(self.class_labels):
            class_dir = os.path.join(self.root_dir, class_label)
            file_names = os.listdir(class_dir)
            for file_name in file_names:
                file_path = os.path.join(class_dir, file_name)
                self.file_paths.append(file_path)
                self.labels.append(i)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        file_path = self.file_paths[index]
        label = self.labels[index]
        spectrogram = np.load(file_path)
        return spectrogram, label


dataset = SpectrogramDataset('../data')
embed()
exit()

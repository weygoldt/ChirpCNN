import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def load_model(modelpath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ChirpCNN().to(device)
    model.load_state_dict(torch.load(modelpath, map_location=device))
    model.eval()
    return model


class SpectrogramDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.class_labels = os.listdir(root_dir)
        self.file_paths = []
        self.labels = []
        self.transform = transform

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
        spectrogram = np.expand_dims(np.load(file_path), axis=0)
        spectrogram = spectrogram.astype("float32")
        spectrogram = torch.from_numpy(spectrogram)
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram, label


class ChirpCNN(nn.Module):
    def __init__(self):
        # SuperInit the nn.Module parent class
        super(ChirpCNN, self).__init__()

        # Note: The input size of the next must always be the
        # output size of the previous layer

        self.conf1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conf2 = nn.Conv2d(6, 16, 5)

        # Note: The input size of the next must always be the
        # output size of the previous layer. Linear layer flattens
        # the 3d tensor into a 1d tensor. So the input size is the
        # product of the output sizes of all dimensions (except the
        # batch dimension) of the previous layer.
        # See https://www.youtube.com/watch?v=pDdP0TFzsoQ
        # for a good explanation.

        self.fc1 = nn.Linear(16 * 29 * 29, 120)
        self.fc2 = nn.Linear(120, 84)

        # Note: The number of output channels of the last layer
        # must be equal to the number of classes

        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Apply first convolutional and pooling layers
        x = self.pool(F.relu(self.conf1(x)))
        x = self.pool(F.relu(self.conf2(x)))

        # Flatten the 3d tensor into a 1d tensor to pass
        # into the fully connected layers. -1 means that
        # the batch size is inferred from the other dimensions

        x = x.view(-1, 16 * 29 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

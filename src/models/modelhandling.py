import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def load_model(modelpath, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = model().to(device)
    mod.load_state_dict(torch.load(modelpath, map_location=device))
    mod.eval()
    return mod


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


class ChirpNet(nn.Module):
    def __init__(self):
        super(ChirpNet, self).__init__()

        # original img size of 128x128

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)

        # after first conv layer: (img_width - kernel_size + 2*padding)/stride + 1
        # that makes (128 - 5 + 2*0) / 1 + 1 = 124

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # after first pooling layer: (124 - 2) / 2 + 1 = 62

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)

        # after second conv layer: (62 - 5 + 2*0) / 1 + 1 = 58
        # the forward pools again so after second pooling layer: (58 - 2) / 2 + 1 = 29
        # so the in-feature size is 16 channels * 29 pixels * 29 pixels

        self.fc1 = nn.Linear(in_features=32 * 29 * 29, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=2)

    def forward(self, x):
        # Apply first convolutional and pooling layers

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the 3d tensor into a 1d tensor to pass
        # into the fully connected layers. -1 means that
        # the batch size is inferred from the other dimensions

        x = x.view(-1, 32 * 29 * 29)

        # call firt fully connected layer
        # apply relu activation function

        x = F.relu(self.fc1(x))

        # call second fully connected layer

        x = F.relu(self.fc2(x))

        # call third fully connected layer
        # no activation function here
        # because we will use cross entropy loss
        # and it applies softmax internally

        x = self.fc3(x)
        return x


class ChirpNet2(nn.Module):
    def __init__(self):
        super(ChirpNet2, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=5, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1
        )
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(
            in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1
        )
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 10 * 10, 2)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = output.view(-1, 24 * 10 * 10)
        output = self.fc1(output)

        return output

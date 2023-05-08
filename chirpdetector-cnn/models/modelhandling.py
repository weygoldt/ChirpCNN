import os

import numpy as np
import torch
from torch.utils.data import Dataset
from utils.logger import make_logger

logger = make_logger(__name__)


def check_device():
    if torch.cuda.is_available() is True:
        device = torch.device("cuda")  # nvidia / amd gpu
    elif torch.backends.mps.is_available() is True:
        device = torch.device("mps")  # apple m1 gpu
    else:
        device = torch.device("cpu")  # no gpu
    return device


device = check_device()


def load_model(modelpath, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mod = model().to(device)
    mod.load_state_dict(torch.load(modelpath, map_location=device))
    mod.eval()
    return mod


def train_epoch(model, train_dl, optimizer, criterion, scheduler):
    train_loss, correct_prediction = 0.0, 0.0
    model.train()
    for inputs, labels in train_dl:
        inputs, labels = inputs.to(device), labels.to(device)

        # TODO: normalize the inputs
        # normalize the inputs
        # inputs_m, inputs_s = inputs.mean(), inputs.std()
        # inputs = (inputs - inputs_m) / inputs_s

        # Zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Keep stats for Loss and Accuracy
        train_loss += loss.item() * inputs.size(0)

        # get predicted class
        _, prediction = torch.max(outputs, 1)

        # count predictions that match the target label
        correct_prediction += torch.sum(prediction == labels).item()

    return train_loss, correct_prediction


def validate_epoch(model, val_dl, criterion):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for images, labels in val_dl:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            valid_loss += loss.item() * images.size(0)
            _, predictions = torch.max(output.data, 1)
            val_correct += (predictions == labels).sum().item()

    return valid_loss, val_correct, output, labels


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
        spectrogram = np.expand_dims(np.load(file_path), axis=0)
        spectrogram = spectrogram.astype("float32")
        spectrogram = torch.from_numpy(spectrogram)
        return spectrogram, label

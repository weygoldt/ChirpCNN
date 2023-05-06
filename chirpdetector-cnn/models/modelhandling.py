import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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


def training(model, train_dl, num_epochs):
    # Loss Function, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.001,
        steps_per_epoch=int(len(train_dl)),
        epochs=num_epochs,
        anneal_strategy="linear",
    )

    # Repeat for each epoch
    loss_tracker = []
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        # Repeat for each batch in the training set
        for i, data in enumerate(train_dl):
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Keep stats for Loss and Accuracy
            running_loss += loss.item()

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

            if i % 10 == 0:  # print every 10 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / 10)
                )
            loss_tracker.append(loss.cpu().detach().numpy())

        # Print stats at the end of the epoch
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f"Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}")

    print("Finished Training")
    plt.plot(loss_tracker)
    plt.title("training loss")
    plt.xlabel("batch")
    plt.show()


def inference(model, val_dl):
    correct_prediction = 0
    total_prediction = 0

    # Disable gradient updates
    with torch.no_grad():
        for data in val_dl:
            # Get the input features and target labels, and put them on the GPU
            inputs, labels = data[0].to(device), data[1].to(device)

            # Normalize the inputs
            inputs_m, inputs_s = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_m) / inputs_s

            # Get predictions
            outputs = model(inputs)

            # Get the predicted class with the highest score
            _, prediction = torch.max(outputs, 1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == labels).sum().item()
            total_prediction += prediction.shape[0]

    acc = correct_prediction / total_prediction
    print(f"Accuracy: {acc:.2f}, Total items: {total_prediction}")


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

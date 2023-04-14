#!/usr/bin/env python3

import argparse
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from tqdm import tqdm

from models.modelhandling import ChirpNet, ChirpNet2, SpectrogramDataset
from utils.filehandling import ConfLoader
from utils.logger import make_logger
from utils.plotstyle import PlotStyle

ps = PlotStyle()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
conf = ConfLoader("config.yml")
logger = make_logger(__name__)


# create a function (this my favorite choice)
def RMSELoss(predicted, target):
    return torch.sqrt(torch.mean((predicted - target) ** 2))


def viz(dataloader, classes, save=False, path="dataset.png"):
    rows, cols = 5, 10
    fig, axs = plt.subplots(
        rows, cols, figsize=(24 * ps.cm, 12 * ps.cm), constrained_layout=True
    )
    for ax in axs.flat:
        spectrogram, label = next(iter(dataloader))
        ax.imshow(spectrogram[0, 0, :, :], origin="lower", interpolation="none")
        ax.set_title(classes[label[0]], loc="center", fontsize=10)
        ax.axis("off")
    if save:
        logger.info(f"Saving dataset plot to {path}")
        plt.savefig(path, dpi=300)
        plt.close()
        plt.clf()
        plt.cla()
    # plt.show()


def main():
    save = True
    # criterion = RMSELoss

    # Initialize dataset and set up dataloaders
    dataset = SpectrogramDataset(conf.training_data_path)
    classes = dataset.class_labels
    train_size = int(conf.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Initialize model, loss, and optimizer
    num_epochs = conf.num_epochs
    batch_size = conf.batch_size
    learning_rate = conf.learning_rate

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    logger.info(f"Classes: {classes}")
    logger.info(f"Labels: {np.arange(len(classes))}")
    viz(train_loader, classes, save=True, path=conf.plot_dir + "/dataset.png")

    model = ChirpNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    n_iterations = math.ceil(n_total_steps / batch_size)
    logger.info(f"Number of steps per epoch: {n_total_steps}")
    logger.info(f"Number of iterations: {n_iterations}")

    # Train the model
    step_loss = []
    for epoch in range(num_epochs):
        model.train()
        for i, (spectrograms, labels) in enumerate(train_loader):
            # Get data and label and put them on the GPU
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(spectrograms)  # forward pass
            loss = criterion(outputs, labels)  # calculate the loss

            # Backward and optimize
            optimizer.zero_grad()  # zero the gradient buffers
            loss.backward()  # calculate the gradients
            optimizer.step()  # update the weights

            step_loss.append(loss.item())

            if (i + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}"
                )

    logger.info("Finished training")

    # Test the model
    validation_loss = []
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]

        model.eval()

        for spectrograms, labels in tqdm(test_loader, desc="Testing"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            outputs = model(spectrograms)

            # Calculate loss
            loss = criterion(outputs, labels)
            validation_loss.append(loss.item())

            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                # Depending on the size of the dataset and the
                # batch size, the last batch sometimes contains
                # less than batch_size samples. In this case,
                # we just skip the last batch. I will fix this once
                # I have a better and bigger dataset.

                try:
                    label = labels[i]
                    pred = predicted[i]
                except:
                    continue

                if label == pred:
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        logger.info(f"Accuracy of the network: {acc} %")

        for i in range(len(classes)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            logger.info(f"Accuracy of {classes[i]}: {acc} %")

    # Save the model
    if save:
        logger.info(f"Saving model to {conf.save_dir}")
        torch.save(model.state_dict(), conf.save_dir)

    fig, ax = plt.subplots()
    ax.plot(step_loss, label="train_loss")
    ax.plot(validation_loss, label="val_loss")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()

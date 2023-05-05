#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.audioclassifier import AudioClassifier
from models.modelhandling import (
    SpectrogramDataset,
    check_device,
    inference,
    training,
)
from torch.utils.data import DataLoader
from utils.filehandling import ConfLoader
from utils.logger import make_logger
from utils.plotstyle import PlotStyle

ps = PlotStyle()
device = check_device()
conf = ConfLoader("config.yml")
logger = make_logger(__name__)


def viz(dataloader, classes, save=False, path="dataset.png"):
    rows, cols = 10, 10
    fig, axs = plt.subplots(
        rows, cols, figsize=(24 * ps.cm, 24 * ps.cm), constrained_layout=True
    )
    for ax in axs.flat:
        spectrogram, label = next(iter(dataloader))
        ax.imshow(spectrogram[0, 0, :, :], origin="lower", interpolation="none")
        ax.set_title(classes[label[0]], loc="center", fontsize=10)
        ax.axis("off")
    if save:
        logger.info(f"Saving dataset plot to {path}")
        plt.savefig(path, dpi=300)
        plt.close("all")
        plt.close(fig)
        plt.clf()
        plt.cla()


def main():
    save = True

    # Initialize dataset and set up dataloaders
    dataset = SpectrogramDataset(conf.training_data_path)
    classes = dataset.class_labels
    logger.info(f"Classes: {classes}")
    logger.info(f"Labels: {np.arange(len(classes))}")

    # Divide dataset into train and test set
    train_size = int(conf.train_size * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    # Initialize model, loss, and optimizer
    num_epochs = conf.num_epochs
    batch_size = conf.batch_size
    learning_rate = conf.learning_rate

    # Create dataloaders for the train and test set
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # Visualize a few examples from the dataset
    viz(train_loader, classes, save=True, path=conf.plot_dir + "/dataset.png")

    model = AudioClassifier().to(device)

    # train the model
    training(model, train_loader, num_epochs)

    # test the model
    inference(model, test_loader, classes)

    # Save the model
    if save:
        logger.info(f"Saving model to {conf.save_dir}")
        torch.save(model.state_dict(), conf.save_dir)


if __name__ == "__main__":
    main()

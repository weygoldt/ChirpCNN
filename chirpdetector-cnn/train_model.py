#!/usr/bin/env python3

"""
Uses data saved in the training data path specified in the conf.yml to train 
the convolutional neural network to detect chirps.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from IPython import embed
from models.audioclassifier import AudioClassifier
from models.modelhandling import (
    SpectrogramDataset,
    check_device,
    train_epoch,
    validate_epoch,
)
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, SubsetRandomSampler
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
        ax.imshow(
            spectrogram[0, 0, :, :],
            origin="lower",
            interpolation="none",
            aspect="equal",
        )
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
    # Initialize dataset and set up dataloaders
    dataset = SpectrogramDataset(conf.training_data_path)
    classes = dataset.class_labels
    logger.info(f"Classes: {classes}")
    logger.info(f"Labels: {np.arange(len(classes))}")

    # Initialize model, loss, and optimizer
    num_epochs = conf.num_epochs
    batch_size = conf.batch_size
    learning_rate = conf.learning_rate
    kfolds = conf.kfolds
    torch.manual_seed(42)
    splits = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    foldperf = {}
    for fold, (train_idx, val_idx) in enumerate(
        splits.split(np.arange(len(dataset)))
    ):
        print("Fold {}".format(fold + 1))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=train_sampler
        )
        test_loader = DataLoader(
            dataset, batch_size=batch_size, sampler=test_sampler
        )
        viz(
            train_loader,
            classes,
            save=True,
            path=conf.plot_dir + f"/fold_{fold}.png",
        )

        model = AudioClassifier().to(device)

        # Loss Function, Optimizer and Scheduler
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            steps_per_epoch=int(len(train_loader)),
            epochs=num_epochs,
            anneal_strategy="linear",
        )

        for epoch in range(num_epochs):
            train_loss, train_correct = train_epoch(
                model=model,
                train_dl=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            test_loss, test_correct = validate_epoch(
                model=model,
                val_dl=test_loader,
                criterion=criterion,
            )

            train_loss = train_loss / len(train_loader.sampler)
            train_acc = train_correct / len(train_loader.sampler) * 100
            test_loss = test_loss / len(test_loader.sampler)
            test_acc = test_correct / len(test_loader.sampler) * 100

            print(
                "Epoch:{}/{} AVG Training Loss:{:.3f} AVG Test Loss:{:.3f} AVG Training Acc {:.2f} % AVG Test Acc {:.2f} %".format(
                    epoch + 1,
                    num_epochs,
                    train_loss,
                    test_loss,
                    train_acc,
                    test_acc,
                )
            )
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)

        foldperf["fold{}".format(fold + 1)] = history

    avg_train_loss = np.mean(history["train_loss"])
    avg_test_loss = np.mean(history["test_loss"])
    avg_train_acc = np.mean(history["train_acc"])
    avg_test_acc = np.mean(history["test_acc"])

    print("Performance of {} fold cross validation".format(kfolds))
    print(
        "Average Training Loss: {:.4f} \t Average Test Loss: {:.4f} \t Average Training Acc: {:.3f} \t Average Test Acc: {:.3f}".format(
            avg_train_loss, avg_test_loss, avg_train_acc, avg_test_acc
        )
    )

    logger.info(f"Saving model to {conf.save_dir}")
    torch.save(model.state_dict(), conf.save_dir)

    testl_f, tl_f, testa_f, ta_f = [], [], [], []
    for f in range(1, kfolds + 1):
        tl_f.append(np.mean(foldperf["fold{}".format(f)]["train_loss"]))
        testl_f.append(np.mean(foldperf["fold{}".format(f)]["test_loss"]))
        ta_f.append(np.mean(foldperf["fold{}".format(f)]["train_acc"]))
        testa_f.append(np.mean(foldperf["fold{}".format(f)]["test_acc"]))

    print("Performance of {} fold cross validation".format(kfolds))
    print(
        "Average Training Loss: {:.3f} \t Average Test Loss: {:.3f} \t Average Training Acc: {:.2f} \t Average Test Acc: {:.2f}".format(
            np.mean(tl_f), np.mean(testl_f), np.mean(ta_f), np.mean(testa_f)
        )
    )

    diz_ep = {
        "train_loss_ep": [],
        "test_loss_ep": [],
        "train_acc_ep": [],
        "test_acc_ep": [],
    }

    # TODO: Clean up this mess

    for i in range(num_epochs):
        diz_ep["train_loss_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["train_loss"][i]
                    for f in range(kfolds)
                ]
            )
        )
        diz_ep["test_loss_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["test_loss"][i]
                    for f in range(kfolds)
                ]
            )
        )
        diz_ep["train_acc_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["train_acc"][i]
                    for f in range(kfolds)
                ]
            )
        )
        diz_ep["test_acc_ep"].append(
            np.mean(
                [
                    foldperf["fold{}".format(f + 1)]["test_acc"][i]
                    for f in range(kfolds)
                ]
            )
        )
    # Plot losses
    fig, ax = plt.subplots(1, 2, figsize=(20, 10), contstrained_layout=True)
    ax[0].semilogy(diz_ep["train_loss_ep"], label="Train")
    ax[0].semilogy(diz_ep["test_loss_ep"], label="Test")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[0].set_title("CNN loss")

    # Plot accuracies
    ax[1].semilogy(diz_ep["train_acc_ep"], label="Train")
    ax[1].semilogy(diz_ep["test_acc_ep"], label="Test")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    ax[1].set_title("CNN accuracy")
    plt.savefig("../testing_data/losses.png")
    plt.show()

    # TODO: Add ROC curve


if __name__ == "__main__":
    main()

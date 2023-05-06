#!/usr/bin/env python3
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
    splits = KFold(n_splits=kfolds, shuffle=True, random_state=42)
    history = {
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

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
            embed()

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

    # viz(train_loader, classes, save=True, path=conf.plot_dir + "/dataset.png")

    # Save the model

    embed()
    logger.info(f"Saving model to {conf.save_dir}")
    torch.save(model.state_dict(), conf.save_dir)


if __name__ == "__main__":
    main()

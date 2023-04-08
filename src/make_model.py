#!/usr/bin/env python3

import os
import math 

import numpy as np
import matplotlib.pyplot as plt
from IPython import embed   
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
import torch.nn.functional as F

from utils.plotstyle import PlotStyle
from models.modelhandling import ChirpCNN, SpectrogramDataset

ps = PlotStyle()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def viz(dataloader, classes, save=False, path="dataset.png"):
    rows, cols = 5, 5
    fig, axs = plt.subplots(
            rows, 
            cols, 
            figsize=(10, 10),
            constrained_layout = True
    )
    for ax in axs.flat:
        spectrogram, label = next(iter(dataloader))
        ax.imshow(spectrogram[0, 0, :, :], origin='lower', cmap="magma")
        ax.set_title(classes[label[0]], loc = "center")
        ax.axis('off')
    if save: 
        plt.savefig(path, dpi=300)
    # plt.show()


def main():

    # Initialize dataset and set up dataloaders
    dataset = SpectrogramDataset('../data')
    classes = dataset.class_labels 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Initialize model, loss, and optimizer
    num_epochs = 10
    batch_size = 20
    learning_rate = 0.001

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Classes: {classes}")
    print(f"Labels: {np.arange(len(classes))}")
    viz(train_loader, classes, save=True, path='../assets/dataset.png')
    # exit()

    model = ChirpCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    n_total_steps = len(train_loader)
    n_iterations = math.ceil(n_total_steps / batch_size)
    print(f'Number of steps per epoch: {n_total_steps}')
    print(f'Number of iterations: {n_iterations}')

    # Train the model 
    for epoch in range(num_epochs):
        for i, (spectrograms, labels) in enumerate(train_loader):
            
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass 
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

    print("Finished training")

    # Test the model
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(len(classes))]
        n_class_samples = [0 for i in range(len(classes))]

        for spectrograms, labels in tqdm(test_loader, desc="Testing"):
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)
            outputs = model(spectrograms)

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

                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        
        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network: {acc} %")

        for i in range(len(classes)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of {classes[i]}: {acc} %")

    # Save the model
    torch.save(model.state_dict(), 'models/ChirpCNN.pth')


if __name__ == "__main__":
    main()

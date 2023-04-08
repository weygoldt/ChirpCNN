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
    plt.show()


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
        spectrogram = spectrogram.astype('float32')
        spectrogram = torch.from_numpy(spectrogram)
        if self.transform:
            spectrogram = self.transform(spectrogram)
        return spectrogram, label


class ConvNet(nn.Module):
    def __init__(self):
        
        # SuperInit the nn.Module parent class 
        super(ConvNet, self).__init__()
        
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


def main():
    # Initialize dataset and set up dataloaders
    dataset = SpectrogramDataset('../data')
    classes = dataset.class_labels 
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Initialize model, loss, and optimizer
    num_epochs = 10
    batch_size = 10
    learning_rate = 0.001

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print(f"Classes: {classes}")
    print(f"Labels: {np.arange(len(classes))}")
    viz(train_loader, classes, save=True, path='../assets/dataset.png')

    model = ConvNet().to(device)
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
                try:
                    label = labels[i]
                    pred = predicted[i]
                except:
                    continue

                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
        
        print(np.shape(n_class_correct))
        print(np.shape(n_class_samples))

        acc = 100.0 * n_correct / n_samples
        print(f"Accuracy of the network: {acc} %")

        for i in range(len(classes)):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"Accuracy of {classes[i]}: {acc} %")

        embed()
        exit()


if __name__ == "__main__":
    main()

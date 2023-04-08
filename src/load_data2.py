import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed   
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor

from utils.plotstyle import PlotStyle

ps = PlotStyle()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Define a custom dataset class
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
        spectrogram = np.load(file_path)

        if self.transform:
            spectrogram = self.transform(spectrogram)

        return spectrogram, label


def viz(dataloader):
    rows, cols = 5, 5
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    for ax in axs.flat:
        spectrogram, label = next(iter(dataloader))
        ax.imshow(spectrogram[0, :, :], origin='lower', cmap="gray")
        ax.set_title(f'label: {label[0]}')
        ax.axis('off')
    plt.savefig('spectrogram.png', dpi=300)
    plt.show()



dataset = SpectrogramDataset('../data')
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

viz(train_loader)

embed()
exit()

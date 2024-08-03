from tqdm import tqdm 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import numpy as np
import random
import os

### EMNIST Somehow now working?

sequence_length = 5

class SequenceDigitDataset(Dataset):
    def __init__(self, dataset, sequence_length=5, transform=None):
        self.dataset = dataset
        self.sequence_length = sequence_length
        self.transform = transform

    def __len__(self):
        return len(self.dataset) // self.sequence_length

    def __getitem__(self, idx):
        sequence_images = []
        sequence_labels = []
        
        for i in range(self.sequence_length):
            img, label = self.dataset[idx * self.sequence_length + i]
            sequence_images.append(img)
            sequence_labels.append(label)
        
        # Concatenate images horizontally
        sequence_image = torch.cat(sequence_images, dim=2)  # Concatenate along width
        
        sequence_labels = torch.tensor(sequence_labels)
        
        return sequence_image, sequence_labels


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if not os.path.exists('./data/EMNIST'):
    os.makedirs('./data/EMNIST')

if not os.path.exists('./data/MNIST'):
    os.makedirs('./data/MNIST')

def load_data(create_sequence_data=False):

    train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    if create_sequence_data == True:
        train_dataset = SequenceDigitDataset(train_mnist, sequence_length=sequence_length, transform=transform)
        test_dataset = SequenceDigitDataset(test_mnist, sequence_length=sequence_length, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    else:
        train_loader = DataLoader(train_mnist, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_mnist, batch_size=32, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    train_loader, test_loader = load_data()
    for images, labels in train_loader:
        print(images.shape, labels.shape)
        break
from tqdm import tqdm 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import numpy as np
import random
import os

### EMNIST Somehow now working?

class SequenceMNIST(Dataset):
    def __init__(self, mnist_data, transform=None, sequence_length=5):
        self.mnist_data = mnist_data
        self.transform = transform
        self.sequence_length = sequence_length
    
    def __len__(self):
        return len(self.mnist_data)
    
    def __getitem__(self, idx):
        images, labels = [], []
        for _ in range(self.sequence_length):
            img, label = self.mnist_data[random.randint(0, len(self.mnist_data) - 1)]
            if self.transform:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
        images = torch.stack(images, dim=0) # (sequence_length, 1, 28, 28)
        return images, torch.tensor(labels, dtype=torch.long) 

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if not os.path.exists('./data/EMNIST'):
    os.makedirs('./data/EMNIST')

if not os.path.exists('./data/MNIST'):
    os.makedirs('./data/MNIST')

train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=None)
test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=None)

def load_data(sequence_length=5):
    train_dataset = SequenceMNIST(train_mnist, transform=transform, sequence_length=sequence_length)
    test_dataset = SequenceMNIST(test_mnist, transform=transform, sequence_length=sequence_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    return train_loader, test_loader

train_loader, test_loader = load_data()
from tqdm import tqdm 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

### EMNIST Somehow now working?

class SequenceMNIST(Dataset):
    def __init__(self, mnist_dataset, seq_len=5, transform=None):
        self.mnist_dataset = mnist_dataset
        self.seq_len = seq_len
        self.transform = transform

    def __len__(self):
        return len(self.mnist_dataset) // self.seq_len

    def __getitem__(self, idx):
        images = []
        labels = []
        for i in range(self.seq_len):
            img, label = self.mnist_dataset[idx * self.seq_len + i]
            if self.transform:
                img = self.transform(img)
            images.append(img)
            labels.append(label)
        images = torch.stack(images)  # shape (seq_len, 1, 28, 28)
        labels = torch.tensor(labels)  # shape (seq_len)
        return images, labels

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if not os.path.exists('./data/EMNIST'):
    os.makedirs('./data/EMNIST')

if not os.path.exists('./data/MNIST'):
    os.makedirs('./data/MNIST')

if not os.path.exists('./data/sequence/MNIST'):
    os.makedirs('./data/sequence/MNIST')

def show_sequence(images, labels):
    fig, axes = plt.subplots(nrows=1, ncols=len(images), figsize=(10, 2))
    for i, (img, label) in enumerate(zip(images, labels)):
        axes[i].imshow(img.squeeze(), cmap='gray')
        axes[i].set_title(f"Label: {label.item()}")
        axes[i].axis('off')
    plt.show()

def load_data(create_sequence_data=False):

    train_mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_mnist = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_mnist, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_mnist, batch_size=32, shuffle=False)

    return train_loader, test_loader

def load_data_sequence():
    
    train_set = datasets.MNIST(root='./data/sequence', train=True, download=True, transform=None)
    test_set = datasets.MNIST(root='./data/sequence', train=False, download=True, transform=None)

    train_seq_set = SequenceMNIST(train_set, seq_len=5, transform=transform)
    test_seq_set = SequenceMNIST(test_set, seq_len=5, transform=transform)

    train_loader = DataLoader(train_seq_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_seq_set, batch_size=32, shuffle=False)

    return train_loader, test_loader

if __name__ == '__main__':
    user_input = input("Do you want to download the normal MNIST data (1) or sequenced MNIST data (2)?")
    
    if user_input == "1":
        train_loader, test_loader = load_data()
        for images, labels in train_loader:
            print(images.shape, labels.shape)
            break

    if user_input == "2":
        train_loader, test_loader = load_data_sequence()
        for images, labels in train_loader:
            print(images.shape, labels.shape)
            show_sequence(images[0], labels[0])
            break
from tqdm import tqdm 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import os

### EMNIST Somehow now working?

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if not os.path.exists('./data/EMNIST'):
    os.makedirs('./data/EMNIST')

if not os.path.exists('./data/MNIST'):
    os.makedirs('./data/MNIST')

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data' , train=False, download=True, transform=transform)

def load_data():

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader

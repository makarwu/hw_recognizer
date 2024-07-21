import numpy as np
from tqdm import tqdm
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from layer import Layer
from model import Convolution
from activation import Sigmoid
from reshape import Reshape
from dense import Dense
from activation import binary_cross_entropy, binary_cross_entropy_prime
from keras.datasets import mnist
from keras.utils import to_categorical

def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

track_loss = []

def train(network, loss, loss_prime, x_train, y_train, epochs=1000, learning_rate=1e-2, verbose=True):
    for e in tqdm(range(epochs), desc="Training"):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward pass
            output = predict(network, x)
            # error
            error += loss(y, output)
            # backward pass
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)
        
        error /= len(x_train)
        track_loss.append(error)
        if verbose:
            print(f"{e + 1}/{epochs}, error={error}")

"""def preprocess_data(dataset, limit):
    indices = []
    class_counts = {0: 0, 1: 0}

    for idx, (_, label) in enumerate(dataset):
        if label in class_counts and class_counts[label] < limit:
            indices.append(idx)
            class_counts[label] += 1
        if all(count == limit for count in class_counts.values()):
            break
    
    subset = Subset(dataset, indices)

    loader = DataLoader(subset, batch_size=len(subset), shuffle=True)
    data_iter = iter(loader)
    images, labels = next(data_iter)

    images = images.view(len(images), 1, 28, 28).float() / 255.0
    labels = torch.nn.functional.one_hot(labels, num_classes=2).float().view(len(labels), 2, 1)

    return images, labels

transform = transforms.Compose([transforms.ToTensor()])

if not os.path.exists('./data/MNIST'):
    os.makedirs('./data/MNIST')

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

x_train, y_train = preprocess_data(train_dataset, 100)
x_test, y_test = preprocess_data(test_dataset, 100)"""

def preprocess_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype("float32") / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 2, 1)
    return x, y

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 1000)

print(x_train.shape)  # Should be (200, 1, 28, 28)
print(y_train.shape)  # Should be (200, 2, 1)

network = [
    Convolution((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 2),
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

### PLOTTING THE LOSS ###

plt.figure(figsize=(12, 5))
plt.plot(track_loss, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.show()

### TESTING THE MODEL FROM SCRATCH ###

for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")
    print(f"Accuracy {(np.argmax(output) == np.argmax(y) / len(x_test)) * 100:.2f}")
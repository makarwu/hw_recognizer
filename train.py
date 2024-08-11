import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import HCRM, HSRM
from tqdm import tqdm
from data import load_data, load_data_sequence

# Reinit the data and model everytime you want to train!

user_input = input("What model do you want to train?\n1. Single Digit Model\n2. Sequence Digit Model\nChoose between 1 and 2!\n")

if user_input == "1":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HCRM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(train_loader)

    def validate(model, test_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
        return running_loss / len(test_loader), correct / len(test_loader.dataset)

    train_losses = []
    val_losses = []
    val_accuracies = []
    num_epochs = 10

    train_loader, test_loader = load_data()

    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

    ### PLOTTING LOSSES AND ACCURACIES ###

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()


    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.show()

    ### SAVING THE MODEL ###

    torch.save(model.state_dict(), './model/handwritten_character_recognition_model.pth')

if user_input == "2":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HSRM().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    def train(model, train_loader, criterion, optimizer, device):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # FYI: Flatten the output to (batch_size * seq_len, num_classes)
            outputs = outputs.view(-1, outputs.size(-1)) 
            # FYI: Flatten the labels to (batch_size * seq_len)
            labels = labels.view(-1) 
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        print(f"Train loss:, {train_loss:.4f}")
        return train_loss
    
    def validate(model, test_loader, criterion, device):
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Validating"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.view(-1, outputs.size(-1))
                labels = labels.view(-1)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_loss = running_loss / len(test_loader)
        accuracy = correct / total

        return avg_loss, accuracy

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    num_epochs = 10

    train_loader, test_loader = load_data_sequence()

    for epoch in tqdm(range(num_epochs)):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

    ## PLOTTING LOSSES AND ACCURACIES ###
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')
    plt.legend()

    plt.show()

    ### SAVING THE MODEL ###
    torch.save(model.state_dict(), './model/handwritten_character_recognition_model_lstm.pth')
    
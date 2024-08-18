import torch
import torch.nn as nn 
import torch.nn.functional as F

### IMPLEMENTED WITH PYTORCH ###

class HCRM(nn.Module):
    def __init__(self):
        super(HCRM, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class HSRM(nn.Module):
    def __init__(self, num_classes=10, num_layers=2, hidden_size=256):
        super(HSRM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        self.lstm = nn.LSTM(128 * 7 * 7, hidden_size, num_layers, batch_first=True, dropout=0.5)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        batch_size, seq_len, channels, height, width = x.size()
        print(f"Input shape: {x.shape}")
        x = x.view(batch_size * seq_len, channels, height, width)
        c_out = self.cnn(x)
        print(f"Shape after CNN: {c_out.shape}")
        c_out = c_out.view(batch_size, seq_len, -1)
        print(f"Shape before LSTM: {c_out.shape}")
        r_out, (h_n, c_n) = self.lstm(c_out)
        print(f"Shape after LSTM: {r_out.shape}")
        r_out2 = self.fc_layers(r_out)
        print(f"Final output shape: {r_out2.shape}")
        
        return r_out2
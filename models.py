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
    def __init__(self, num_classes=10, num_layers=1, hidden_size=128):
        super(HSRM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.lstm = nn.LSTM(64 * 7 * 7, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        #print("\nx size:", x.size()) #DEBUG
        batch_size, seq_len, channels, height, width = x.size()
        x = x.view(batch_size * seq_len, channels, height, width)
        c_out = self.cnn(x)
        c_out = c_out.view(batch_size, seq_len, -1)
        r_out, (h_n, c_n) = self.lstm(c_out)
        r_out2 = self.fc(r_out)
        return r_out2
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

"""class HSRM(nn.Module):
    def __init__(self, sequence_length=5, num_classes=10, input_size=28, hidden_size=128, num_layers=2):
        super(HSRM, self).__init__()
        self.sequence_length = sequence_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes * sequence_length)

    def forward(self, x):
        # Reshape input to (batch_size, sequence_length, input_size)
        x = x.squeeze(1)  # Remove channel dimension (batch_size, height, width)
        x = x.permute(0, 2, 1)  # (batch_size, width, height)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x.view(-1, self.sequence_length, 10)"""

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
        batch_size, seq_len, channels, height, width = x.size()
        c_in = x.view(batch_size * seq_len, channels, height, width)
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, seq_len, -1)
        r_out, (h_n, c_n) = self.lstm(r_in)
        r_out2 = self.fc(r_out)
        return r_out2
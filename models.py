import torch.nn as nn 
import torch.nn.functional as F

### IMPLEMENTED WITH PYTORCH ###
class HCRM(nn.Module):
    def __init__(self):
        super(HCRM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.lstm = nn.LSTM(64*7*7, 128, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, 27) # 26 letters + 1 for blank character in CTC

    def forward(self, x):
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.view(cnn_out.size(0), -1, 64*7*7)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        out = self.fc(lstm_out[:, -1, :])
        return out


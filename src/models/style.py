from torch import nn
import torch.nn.functional as F

class StyleNet(nn.Module):
    def __init__(self, sx=16, sy=8):
        super(StyleNet, self).__init__()
        self.asize = sx * sy
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.apool = nn.AdaptiveAvgPool2d((sx, sy))
        self.fc1 = nn.Linear(sx * sy * 16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.apool(F.relu(self.conv2(x)))
        x = x.view(-1, self.asize * 16)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x))
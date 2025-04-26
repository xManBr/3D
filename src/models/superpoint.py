# src/models/superpoint.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperPointNet(nn.Module):
    def __init__(self):
        super(SuperPointNet, self).__init__()
        # Definição básica simplificada
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.detector = nn.Conv2d(128, 65, 1, stride=1, padding=0)
        self.descriptor = nn.Conv2d(128, 256, 1, stride=1, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        det = self.detector(x)
        desc = self.descriptor(x)
        desc = F.normalize(desc, p=2, dim=1)

        return det, desc

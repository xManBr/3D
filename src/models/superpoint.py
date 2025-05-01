# src/models/superpoint.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperPointNet(nn.Module):
    def __init__(self, weights_path=None):
        super(SuperPointNet, self).__init__()

        # Encoder
        self.conv1a = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        # Detector Head
        self.convPa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)  # 65 = 8x8 + 1 "dustbin"

        # Descriptor Head
        self.convDa = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)

        # Carregar pesos oficiais, se fornecido
        if weights_path is not None:
            print(f"📥 Carregando pesos SuperPoint de: {weights_path}")
            self.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print("✅ Pesos carregados com sucesso!")

    def forward(self, x):
        # Shared Encoder
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool(x)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = self.pool(x)

        x = F.relu(self.conv4a(x))
        x = F.relu(self.conv4b(x))

        # Detector Head
        cPa = F.relu(self.convPa(x))
        semi = self.convPb(cPa)

        # Descriptor Head
        cDa = F.relu(self.convDa(x))
        desc = self.convDb(cDa)
        desc = F.normalize(desc, p=2, dim=1)

        return semi, desc

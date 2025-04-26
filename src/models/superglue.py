# src/models/superglue.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperGlueNet(nn.Module):
    def __init__(self):
        super(SuperGlueNet, self).__init__()
        # SuperGlue simplificado para matching direto
        self.linear = nn.Linear(512, 1)

    def forward(self, desc0, desc1):
        # Calcula similaridade entre descritores
        desc0 = F.normalize(desc0, p=2, dim=1)
        desc1 = F.normalize(desc1, p=2, dim=1)
        sim_matrix = torch.einsum('bd,nkd->bnk', desc0, desc1)

        scores = self.linear(sim_matrix)
        return scores

import torch
from torch import nn
import torch.nn.functional as F


class DELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, domain_dir_pred, domain_dir):
        return 1 - torch.einsum("ij,ij->i", domain_dir_pred, domain_dir).mean()
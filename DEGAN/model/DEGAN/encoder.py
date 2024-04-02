import torch
from torch import nn

class DomainEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Linear(1, 1)
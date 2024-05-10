import torch.nn as nn

class BaseLoss(nn.Module):
    def __init__(self, name, mult):
        super().__init__()
        self.name = name
        self.mult = mult
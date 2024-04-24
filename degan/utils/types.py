import torch

class TorchFloat32:
    def __init__(self):
        pass
    def __new__(cls):
        return torch.float32
    
class TorchFloat64:
    def __init__(self):
        pass
    def __new__(cls):
        return torch.float64
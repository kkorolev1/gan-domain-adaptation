import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2

from DEGAN.base import BaseModel

class DomainEncoder(BaseModel):
    def __init__(self, domain_dim):
        super().__init__()
        self.model = self._load_model(domain_dim)

    def _load_model(self, domain_dim):
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, domain_dim)
        return model

    def forward(self, img):
        return self.model(img)
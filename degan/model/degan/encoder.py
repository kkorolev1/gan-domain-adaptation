import torch
from torch import nn
#from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.transforms import v2

from degan.base import BaseModel

# class DomainEncoder(BaseModel):
#     def __init__(self, domain_dim):
#         super().__init__()
#         self.model = self._load_model(domain_dim)

#     def _load_model(self, domain_dim):
#         model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
#         model.fc = nn.Linear(model.fc.in_features, domain_dim)
#         return model

#     def forward(self, img):
#         return self.model(img)

class DomainEncoder(BaseModel):
    def __init__(self, domain_dim):
        super().__init__()
        self.model = self._load_model(domain_dim)

    def _load_model(self, domain_dim):
        model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        model.heads.head = nn.Linear(model.heads.head.in_features, domain_dim)
        return model

    def forward(self, img):
        """
        img: (B, C, 224, 224)
        """
        return self.model(img)
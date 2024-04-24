import torch
from torch import nn
# from torchvision.models import resnet50, ResNet50_Weights
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

        self.encoder = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        self.encoder_dim = self.encoder.heads.head.in_features
        self.encoder.heads = nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(self.encoder_dim, 2 * self.encoder_dim),
            nn.GELU(),
            nn.Linear(2 * self.encoder_dim, 4 * self.encoder_dim),
            nn.GELU(),
            nn.Linear(4 * self.encoder_dim, domain_dim)
        )

    def forward(self, img):
        """
        img: (B, C, 224, 224)
        """
        return self.mlp(self.encoder(img))
import torch
from torch import nn
from torchvision.models import ViT_B_16_Weights
from torchvision.transforms import v2
from enum import Enum
from collections import namedtuple

from degan.base import BaseModel
from degan.model.degan.domain_vit import DomainTransformer


ViTSpec = namedtuple("ViTSpec", ["model_kwargs", "weights"])

class ViTEnum(Enum):
    ViT_B_16 = ViTSpec(
        model_kwargs = {
            "image_size": 224,
            "patch_size": 16,
            "num_layers": 12,
            "num_heads": 12,
            "hidden_dim": 768,
            "mlp_dim": 3072
        },
        weights = ViT_B_16_Weights.DEFAULT
    )
    DEFAULT = ViT_B_16


class DomainEncoder(BaseModel):
    domain_dims_per_resolution = [
        [512, 512],
        [512, 512, 512],
        [512, 512, 512],
        [512, 512, 512],
        [512, 512, 512],
        [512, 256, 256],
        [256, 128, 128],
        [128, 64, 64],
        [64, 32, 32]
    ]

    def __init__(self, vit_spec=ViTEnum.DEFAULT, freeze_transformer=True):
        super().__init__()
        self.vit_spec = vit_spec
        self.freeze_transformer = freeze_transformer

        self.transformer = DomainTransformer(
            **vit_spec.value.model_kwargs,
            domain_dims_per_resolution=DomainEncoder.domain_dims_per_resolution
        )

        self._load_transformer_weights(self.transformer, freeze_transformer)

        self.transform = v2.Compose([
            v2.Resize((224, 224))
        ])

    def _load_transformer_weights(self, transformer, freeze_transformer):
        weights = self.vit_spec.value.weights
        state_dict = weights.get_state_dict(progress=True, check_hash=True)
        transformer.load_state_dict(state_dict, strict=False)

        # Patch for old versions
        for i in range(len(transformer.encoder.layers)):
            prefix = f"encoder.layers.encoder_layer_{i}.mlp."
            for j in range(2):
                for type in ["weight", "bias"]:
                    old_key = f"{prefix}linear_{j+1}.{type}"
                    new_key = f"{prefix}{3*j}.{type}"
                    if old_key in state_dict:
                        state_dict[new_key] = state_dict.pop(old_key)
        
        if freeze_transformer:
            params_to_freeze = state_dict.keys()
            for param_name, param in transformer.named_parameters():
                if param_name in params_to_freeze:
                    param.requires_grad = False

    def forward(self, img):
        """
            img: (B, C, 1024, 1024)
        """
        img = self.transform(img)
        return self.transformer(img)
import torch
import numpy as np
from degan.base.base_metric import BaseMetric

class DomainOffsetNorm(BaseMetric):
    def __init__(self, layer_index=0, **kwargs):
        super().__init__(**kwargs)
        self.layer_index = layer_index

    def __call__(self, domain_offsets, *args, **kwargs):
        return torch.norm(domain_offsets[self.layer_index], dim=1).mean().item()
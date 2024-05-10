import torch
import numpy as np
from degan.base.base_metric import BaseMetric

class DomainOffsetNorm(BaseMetric):
    def __init__(self, chunk_index=0, **kwargs):
        super().__init__(**kwargs)
        self.chunk_index = chunk_index

    def __call__(self, domain_chunks, *args, **kwargs):
        """
        domain_chunks: list of domain offset chunks
        """
        return torch.norm(domain_chunks[self.chunk_index], dim=1).mean().item()
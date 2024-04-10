import torch
from degan.base.base_metric import BaseMetric

class SemanticScore(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, gen_emb, domain_emb):
        """
        gen_emb: (N, D)
        domain_emb: (D,)
        """
        cos_sim = gen_emb @ domain_emb
        return cos_sim.mean().item()
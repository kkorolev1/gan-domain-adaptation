import torch
from degan.base.base_metric import BaseMetric


def get_tril_mask(size):
    mask = torch.ones((size, size), dtype=torch.bool)
    return torch.tril(mask).logical_not()

class DiversityScore(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, gen_emb):
        """
        gen_emb: (N, D)
        """
        cosine_sim = gen_emb @ gen_emb.T
        mask = get_tril_mask(cosine_sim.shape[0])
        return torch.mean(1 - cosine_sim[mask]).item()
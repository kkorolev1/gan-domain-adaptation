import torch
import numpy as np
from degan.base.base_metric import BaseMetric
from degan.utils import get_tril_mask

class DiversityScore(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, gen_emb, *args, **kwargs):
        """
        gen_emb: (N, D)
        """
        if len(gen_emb.shape) == 1 or gen_emb.shape[0] == 1:
            return 0
        cosine_sim = gen_emb @ gen_emb.T
        mask = get_tril_mask(cosine_sim.shape[0])
        return torch.mean(1 - cosine_sim[mask]).item()
    
class MeanDiversityScore(BaseMetric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(self, domain_to_gen_emb, *args, **kwargs):
        """
        domain_path_to_gen_emb: key=path, value=gen_emb (N, D)
        """
        metric = DiversityScore()
        values = []
        for domain in domain_to_gen_emb:
            values.append(metric(domain_to_gen_emb[domain]))
        return np.mean(values)
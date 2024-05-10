import torch
import numpy as np
from degan.base.base_metric import BaseMetric

class SemanticScore(BaseMetric):
    def __init__(self, name=None):
        super().__init__(name)

    def __call__(self, gen_emb, domain_emb, *args, **kwargs):
        """
        gen_emb: (N, D)
        domain_emb: (D,)
        """
        cos_sim = gen_emb @ domain_emb
        return cos_sim.mean().item()
    
class MeanSemanticScore(BaseMetric):
    def __init__(self, name=None, *args, **kwargs):
        super().__init__(name, *args, **kwargs)

    def __call__(self, domain_to_gen_emb, domain_to_domain_emb, *args, **kwargs):
        """
        domain_path_to_gen_emb: key=path, value=gen_emb (N, D)
        domain_path_to_domain_emb: key=path, value=domain_emb (D,)
        """
        metric = SemanticScore()
        values = []
        for domain in domain_to_gen_emb:
            values.append(metric(domain_to_gen_emb[domain], domain_to_domain_emb[domain]))
        return np.mean(values)
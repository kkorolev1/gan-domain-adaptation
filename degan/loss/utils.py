import torch
import torch.nn.functional as F
from degan.utils import get_tril_mask


def cosine_loss(x, y):
    """
        x: (B, D)
        y: (B, D)
    """
    return (1 - F.cosine_similarity(x, y, dim=1)).mean()

def direction_loss(gen_emb, src_emb, domain_emb, src_emb_proj):
    """
        gen_emb: (B, D)
        src_emb: (B, D)
        domain_emb: (B, D)
        src_emb_proj: (B, D)
    """
    return cosine_loss(gen_emb - src_emb, domain_emb - src_emb_proj)

def tt_direction_loss(gen_emb, domain_emb):
    """
        gen_emb: (B, D)
        domain_emb: (B, D)
    """
    mask = get_tril_mask(gen_emb.shape[0])
    gen_delta = (gen_emb[None, :, :] - gen_emb[:, None, :])[mask]
    domain_delta = (domain_emb[None, :, :] - domain_emb[:, None, :])[mask]
    return cosine_loss(gen_delta, domain_delta)

def domain_norm_loss(domain_offset):
    return ((domain_offset - 1) ** 2).sum(dim=1).mean()
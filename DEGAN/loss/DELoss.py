import torch
from torch import nn
import torch.nn.functional as F


class DELoss(nn.Module):
    def __init__(self, mult_direction=1):
        super().__init__()
        self.mult_direction = mult_direction

    def forward(self, gen_emb, src_emb, domain_emb, src_emb_mc):
        domain_dir_pred = gen_emb - src_emb
        domain_dir_pred_norm = torch.norm(domain_dir_pred)

        domain_dir = domain_emb - src_emb_mc
        domain_dir_norm = torch.norm(domain_dir)

        cos_sim = torch.einsum("ij,ij->i", domain_dir_pred, domain_dir) / domain_dir_pred_norm / domain_dir_norm

        loss_direction = 1 - cos_sim.mean()
        return self.mult_direction * loss_direction
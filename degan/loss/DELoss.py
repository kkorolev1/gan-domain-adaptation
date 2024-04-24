import torch
from torch import nn
import torch.nn.functional as F


class DELoss(nn.Module):
    def __init__(self, mult_direction=1, mult_domain_norm=1, mult_indomain_angle=1):
        super().__init__()
        self.mult_direction = mult_direction
        self.mult_domain_norm = mult_domain_norm
        self.mult_indomain_angle = mult_indomain_angle

    def forward(self, d, gen_emb, src_emb, domain_emb, src_emb_mc):
        domain_dir_pred = gen_emb - src_emb
        domain_dir_pred_norm = domain_dir_pred / torch.norm(domain_dir_pred)

        domain_dir = domain_emb - src_emb_mc
        domain_dir_norm = domain_dir / torch.norm(domain_dir)

        cos_sim = torch.einsum("ij,ij->i", domain_dir_pred_norm, domain_dir_norm)

        loss_direction = (1 - cos_sim).sum()
        
        gen_gram = gen_emb @ gen_emb.T
        src_gram = src_emb @ src_emb.T
        loss_indomain_angle = ((gen_gram - src_gram) ** 2).sum()

        #loss_domain_norm = ((d - 1) ** 2).sum()
        # self.mult_domain_norm * loss_domain_norm

        return self.mult_direction * loss_direction + self.mult_indomain_angle * loss_indomain_angle
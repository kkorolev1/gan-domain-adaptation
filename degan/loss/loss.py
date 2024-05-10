import torch
from torch import nn
import torch.nn.functional as F

from degan.loss.utils import direction_loss, tt_direction_loss, domain_norm_loss


class BaseLoss(nn.Module):
    def __init__(self, name, mult):
        super().__init__()
        self.name = name
        self.mult = mult


class DirectionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_emb, src_emb, domain_emb, src_emb_proj, **kwargs):
        return self.mult * direction_loss(gen_emb, src_emb, domain_emb, src_emb_proj)


class TTDirectionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_emb, domain_emb, **kwargs):
        return self.mult * tt_direction_loss(gen_emb, domain_emb)


class DomainNormLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, domain_offset, **kwargs):
        return self.mult * domain_norm_loss(domain_offset)


class CompositeLoss(nn.Module):
    def __init__(self, loss_modules):
        super().__init__()
        self.loss_modules = loss_modules

    @property
    def loss_names(self):
        return [module.name for module in self.loss_modules]

    def forward(self, **kwargs):
        loss_dict = {}
        for module in self.loss_modules:
            loss_dict[module.name] = module(**kwargs)
        loss_dict["loss"] = torch.sum(torch.stack(list(loss_dict.values())))
        return loss_dict
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights

from degan.utils import requires_grad
from degan.loss.base import BaseLoss
from degan.loss.utils import *



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
    
    def forward(self, domain_offsets, **kwargs):
        return self.mult * torch.sum(torch.cat([domain_norm_loss(offset) for offset in domain_offsets], dim=1))


class CLIPResonstructionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_emb, domain_emb, **kwargs):
        return self.mult * clip_reconstruction_loss(gen_emb, domain_emb)

class L2ResonstructionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_img, domain_img, **kwargs):
        return self.mult * F.mse_loss(gen_img, domain_img)

class VGGPerceptualLoss(BaseLoss):
    default_features_pos = [0, 4, 9, 16, 23]

    def __init__(self, name, mult, resize=True, features_pos=None):
        super().__init__(name, mult)

        if features_pos is None:
            features_pos = VGGPerceptualLoss.default_features_pos

        models = []
        for i in range(1, len(features_pos)):
            start = features_pos[i - 1]
            end = features_pos[i]
            model = vgg16(weights=VGG16_Weights.DEFAULT).features[start: end].cuda().eval()
            requires_grad(model, requires=False)
            models.append(model)

        self.models = nn.ModuleList(models)
        self.transform = F.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    def forward(self, gen_img, domain_img, feature_layers=(0, 1, 2, 3), style_layers=tuple(), **kwargs):
        gen_img = (gen_img - self.mean) / self.std
        domain_img = (domain_img - self.mean) / self.std
        if self.resize:
            gen_img = self.transform(gen_img, mode='bilinear', size=(224, 224), align_corners=False)
            domain_img = self.transform(domain_img, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = gen_img
        y = domain_img
        for i, model in enumerate(self.models):
            x = model(x)
            y = model(y)
            if i in feature_layers:
                loss += F.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += F.l1_loss(gram_x, gram_y)
        return self.mult * loss


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
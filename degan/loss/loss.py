import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from functools import partial

from degan.utils import requires_grad, get_tril_mask
from degan.loss.base import BaseLoss


def cosine_loss(x, y):
    """
        x: (B, D)
        y: (B, D)
    """
    return (1 - F.cosine_similarity(x, y, dim=1)).mean()


class DirectionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_emb, src_emb, domain_emb, src_proj_emb, **kwargs):
        """
            gen_emb: (B, D)
            src_emb: (B, D)
            domain_emb: (B, D)
            src_proj_emb: (B, D)
        """
        return self.mult * cosine_loss(gen_emb - src_emb, domain_emb - src_proj_emb)


class TTDirectionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_emb, domain_emb, **kwargs):
        """
            gen_emb: (B, D)
            domain_emb: (B, D)
        """
        if gen_emb.shape[0] == 1:
            return torch.zeros_like(gen_emb).mean()
        mask = get_tril_mask(gen_emb.shape[0])
        gen_delta = (gen_emb[None, :, :] - gen_emb[:, None, :])[mask]
        domain_delta = (domain_emb[None, :, :] - domain_emb[:, None, :])[mask]
        return self.mult * cosine_loss(gen_delta, domain_delta)


class InDomainAngleLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_emb, src_emb, batch_expand_mult, **kwargs):
        """
            gen_emb: (B, D)
            src_emb: (B, D)
        """
        if batch_expand_mult == 1:
            return torch.zeros_like(gen_emb).mean()
    
        mask = torch.block_diag(*[get_tril_mask(batch_expand_mult) for _ in range(gen_emb.shape[0] // batch_expand_mult)])
        gen_gram = (gen_emb @ gen_emb.T)[mask]
        src_gram = (src_emb @ src_emb.T)[mask]

        return self.mult * F.mse_loss(gen_gram, src_gram)


class CLIPResonstructionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_emb, domain_emb, **kwargs):
        """
            gen_emb: (B, D)
            domain_emb: (B, D)
        """
        return self.mult * cosine_loss(gen_emb, domain_emb)


class L2ResonstructionLoss(BaseLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, gen_img, domain_img, **kwargs):
        return self.mult * F.mse_loss(gen_img, domain_img)


class VGGPerceptualLoss(BaseLoss):
    default_features_pos = [0, 4, 9, 16, 23]

    def __init__(self, name, mult, source_key="src_img", target_key="gen_img", resize=True, features_pos=None, use_feature_layers=tuple(), use_style_layers=tuple()):
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
        self.transform = partial(F.interpolate, mode="bilinear", size=(224, 224), align_corners=False)
        self.source_key = source_key
        self.target_key = target_key
        self.use_feature_layers = set(use_feature_layers)
        self.use_style_layers = set(use_style_layers)
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).cuda())
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).cuda())

    def forward(self, **kwargs):
        source = kwargs[self.source_key]
        target = kwargs[self.target_key]
        source = (source - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            source = self.transform(source)
            target = self.transform(target)
        loss = 0.0
        x = source
        y = target
        for i, model in enumerate(self.models):
            x = model(x)
            y = model(y)
            if i in self.use_feature_layers:
                loss += F.l1_loss(x, y)
            if i in self.use_style_layers:
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
        loss = 0.0
        for module in self.loss_modules:
            module_loss = module(**kwargs)
            loss = loss + module_loss
            loss_dict[module.name] = module_loss.detach()
        loss_dict["loss"] = loss
        return loss_dict
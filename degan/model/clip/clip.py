import clip
import torch.nn as nn
from torchvision.transforms import v2

from degan.base import BaseModel

class CLIP(BaseModel):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        model, preprocess = clip.load(model_name)
        self.model = model
        self.preprocess = v2.Compose([
            v2.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0]),
            *preprocess.transforms[:2],
            preprocess.transforms[-1]
        ])
        self.emb_dim = self.model.token_embedding.embedding_dim

    def encode_img(self, img):
        emb = self.model.encode_image(self.preprocess(img)).float()
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb

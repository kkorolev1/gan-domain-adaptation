import os

from torchvision.transforms import v2
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from PIL import Image

from glob import glob
import random
import logging
from tqdm import tqdm

from degan.utils import prepare_device, requires_grad
from degan.metric import SemanticScore, DiversityScore

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def open_image(path):
    return Image.open(path).convert("RGB")

def get_face_image(path):
    return open_image(path).crop((0 * 1024, 0, 1 * 1024, 1024))

def get_domain_image(path):
    return open_image(path).crop((1 * 1024, 0, 2 * 1024, 1024))

def get_adapted_image(path):
    return open_image(path).crop((2 * 1024, 0, 3 * 1024, 1024))

def semantic_score(domain2emb, domain2adapted):
    scores = []
    metric = SemanticScore()
    for domain_name in domain2adapted:
        domain_emb = domain2emb[domain_name]
        adapted_emb = domain2adapted[domain_name]
        scores.append(metric(adapted_emb, domain_emb))
    return np.mean(scores), np.std(scores)

def diversity_score(domain2adapted):
    scores = []
    metric = DiversityScore()
    for domain_name in domain2adapted:
        adapted_emb = domain2adapted[domain_name]
        scores.append(metric(adapted_emb))
    return np.mean(scores), np.std(scores)

def log_metric(mean, std, name):
    print(f"{name}: {mean:.4f} ± {std:.4f}")

@hydra.main(version_base=None, config_path="configs", config_name="eval")
def main(config):
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)
    logger = logging.getLogger("eval")

    device, device_ids = prepare_device(config["gpus"], logger)
    logger.info(f"Device {device} Ids {device_ids}")

    clip_encoder = instantiate(config["clip_encoder"])
    requires_grad(clip_encoder, requires=False)
    logger.info(clip_encoder)
    clip_encoder = clip_encoder.to(device)

    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    im2ten = lambda x: transform(x).unsqueeze(0).to(device)

    eval_dir = config["eval_dir"]

    domain2emb = {}
    domain2adapted = {}
    logging.info(f"Evaluating {eval_dir}")

    for domain_dir in tqdm(os.listdir(eval_dir)):
        domain_emb = None
        adapted_emb_stack = []
        for filename in glob(os.path.join(eval_dir, domain_dir, "*.png")):
            if domain_emb is None:
                domain_emb = clip_encoder.encode_img(im2ten(get_domain_image(filename))).squeeze(0)
            adapted_emb = clip_encoder.encode_img(im2ten(get_adapted_image(filename))).squeeze(0)
            adapted_emb_stack.append(adapted_emb)
        adapted_emb_stack = torch.stack(adapted_emb_stack, dim=0)
        
        domain2emb[domain_dir] = domain_emb
        domain2adapted[domain_dir] = adapted_emb_stack

    log_metric(*semantic_score(domain2emb, domain2adapted), "SemanticScore")
    log_metric(*diversity_score(domain2adapted), "DiversityScore")

if __name__ == '__main__':
    main()

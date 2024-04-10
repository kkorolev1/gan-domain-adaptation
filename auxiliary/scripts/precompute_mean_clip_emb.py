import warnings
import sys
import os

import numpy as np
import torch

import logging
import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from degan.utils import prepare_device, requires_grad

warnings.filterwarnings("ignore", category=UserWarning)
    
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@torch.no_grad()
def generate_mean_clip_emb(generator, clip_encoder, batch_size, n_samples, device):
    assert n_samples % batch_size == 0, "Number of samples must be divisible by a batch size"
    n_iters = n_samples // batch_size
    mean_emb = torch.zeros(clip_encoder.emb_dim, device=device)
    for i in tqdm.trange(n_iters):
        latents_mc = torch.randn(batch_size, generator.style_dim, requires_grad=False, device=device)
        src_image_mc = generator([latents_mc])[0]
        mean_emb = mean_emb + clip_encoder.encode_img(src_image_mc).sum(dim=0)
    return mean_emb / n_samples

@hydra.main(version_base=None, config_path="../../configs", config_name="precompute")
def main(config):
    OmegaConf.resolve(config)
    config = OmegaConf.to_container(config)

    logger = logging.getLogger("precompute")

    pretrained_cfg = config["pretrained"]

    generator = instantiate(config["generator"])
    generator.load_state_dict(torch.load(pretrained_cfg["generator"])["g_ema"])
    requires_grad(generator, requires=False)
    logger.info(generator)

    clip_encoder = instantiate(config["clip_encoder"])
    requires_grad(clip_encoder, requires=False)
    logger.info(clip_encoder)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"], logger)
    logger.info(f"Device {device} Ids {device_ids}")
    generator = generator.to(device)
    clip_encoder = clip_encoder.to(device)

    mean_emb = generate_mean_clip_emb(
        generator, clip_encoder, 
        batch_size=config["batch_size"], 
        n_samples=config["n_samples"],
        device=device
    )
    torch.save(mean_emb, config["save_path"])


if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
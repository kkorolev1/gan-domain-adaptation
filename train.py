import warnings
import sys
import os

import numpy as np
import torch

import logging

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from degan.trainer import Trainer
from degan.utils import prepare_device, requires_grad
from degan.utils.object_loading import get_dataloaders
from degan.utils.parse_config import ConfigParser


warnings.filterwarnings("ignore", category=UserWarning)
    
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config):
    OmegaConf.resolve(config)
    config = ConfigParser(OmegaConf.to_container(config), resume=config.get("resume", None))

    logger = logging.getLogger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    pretrained_cfg = config["pretrained"]

    generator = instantiate(config["generator"])
    generator.load_state_dict(torch.load(pretrained_cfg["generator"])["g_ema"])
    requires_grad(generator, requires=False)
    logger.info(generator)

    domain_encoder = instantiate(config["domain_encoder"])
    logger.info(domain_encoder)

    clip_encoder = instantiate(config["clip_encoder"])
    requires_grad(clip_encoder, requires=False)
    logger.info(clip_encoder)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["gpus"], logger)
    logger.info(f"Device {device} Ids {device_ids}")
    generator = generator.to(device)
    domain_encoder = domain_encoder.to(device)
    clip_encoder = clip_encoder.to(device)
    
    # get function handles of loss and metrics
    loss_module = instantiate(config["loss"]).to(device)
    metrics = [
        instantiate(metric_dict)
        for metric_dict in config["metrics"]
    ]
    
    logger.info(f'Len epoch {config["trainer"]["len_epoch"]}')
    logger.info(f'Epochs {config["trainer"]["epochs"]}')
    logger.info(f'Train size {len(dataloaders["train"].dataset)}')
    logger.info(f'Val size {len(dataloaders["val"].dataset)}')
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    trainable_params_encoder = filter(
        lambda p: p.requires_grad, domain_encoder.parameters())
    optimizer_encoder = instantiate(
        config["optimizer_encoder"], trainable_params_encoder)
    lr_scheduler_encoder = instantiate(config["lr_scheduler_encoder"], optimizer=optimizer_encoder)
    if "warmup_scheduler" in config:
        lr_scheduler_encoder = instantiate(config["warmup_scheduler"], optimizer=optimizer_encoder, lr_scheduler=lr_scheduler_encoder)
    ema = None
    if "ema" in config:
        ema = instantiate(config["ema"], model=domain_encoder, include_online_model=False)
    trainer = Trainer(
        generator,
        domain_encoder,
        clip_encoder,
        loss_module,
        metrics,
        optimizer_encoder,
        lr_scheduler_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        ema=ema,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()

if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
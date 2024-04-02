import warnings
import sys
import os

import numpy as np
import torch

import logging
import itertools

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from DEGAN.trainer import Trainer
from DEGAN.utils import prepare_device
from DEGAN.utils.object_loading import get_dataloaders
from DEGAN.utils.parse_config import ConfigParser


warnings.filterwarnings("ignore", category=UserWarning)
    
# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
np.random.seed(SEED)


@hydra.main(version_base=None, config_path="DEGAN/configs", config_name="train")
def main(config):
    OmegaConf.resolve(config)
    config = ConfigParser(OmegaConf.to_container(config))

    logger = logging.getLogger("train")

    # setup data_loader instances
    dataloaders = get_dataloaders(config)

    generator = instantiate(config["generator"])
    logger.info(generator)

    encoder = instantiate(config["encoder"])

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"], logger)
    logger.info(f"Device {device} Ids {device_ids}")
    generator = generator.to(device)
    encoder = encoder.to(device)

    # get function handles of loss and metrics
    loss_module = instantiate(config["loss"]).to(device)
    metrics = [
        instantiate(metric_dict)
        for metric_dict in config["metrics"]
    ]

    logger.info(f'Len epoch {config["trainer"]["len_epoch"]}')
    logger.info(f'Epochs {config["trainer"]["epochs"]}')
    logger.info(f'Dataset size {len(dataloaders["train"].dataset)}')
    # build optimizer, learning rate scheduler. delete every line containing lr_scheduler for
    # disabling scheduler
    # optimizer_d = instantiate(
    #     config["optimizer_d"], trainable_params_d)
    trainable_params_encoder = filter(
        lambda p: p.requires_grad, encoder.parameters())
    optimizer_encoder = instantiate(
        config["optimizer_encoder"], trainable_params_encoder)
    lr_scheduler_encoder = instantiate(config["lr_scheduler_encoder"], optimizer_encoder)
    trainer = Trainer(
        generator,
        encoder,
        loss_module,
        metrics,
        optimizer_encoder,
        lr_scheduler_encoder,
        config=config,
        device=device,
        dataloaders=dataloaders,
        len_epoch=config["trainer"].get("len_epoch", None)
    )

    trainer.train()

if __name__ == "__main__":
    sys.argv.append("hydra.job.chdir=False")
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
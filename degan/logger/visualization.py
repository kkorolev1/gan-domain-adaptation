from enum import Enum

from .wandb import WanDBWriter


class VisualizerBackendType(str, Enum):
    tensorboard = "tensorboard"
    wandb = "wandb"


def get_visualizer(config, logger, backend: VisualizerBackendType):
    if backend == VisualizerBackendType.wandb:
        return WanDBWriter(config, logger)

    return None
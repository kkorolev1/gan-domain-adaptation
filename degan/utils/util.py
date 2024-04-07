import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
from torchvision.transforms import v2
from PIL import Image
from pynvml import *


ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def find_device():
    nvmlInit()
    device_count = nvmlDeviceGetCount()
    infos = []

    for i in range(device_count):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        infos.append((i, info.free))

    infos.sort(key=lambda x: -x[1])
    device = infos[0][0]

    nvmlShutdown()

    return device


def prepare_device(n_gpu_use, logger):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    logger.info(f"Num GPUs query: {n_gpu_use}")
    logger.info(f"Num GPUs available: {n_gpu}")
    if n_gpu_use > 0 and n_gpu == 0:
        logger.info(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        logger.info(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    if n_gpu_use == 1:
        id = find_device()
        device = torch.device(f"cuda:{id}")
        list_ids = [id]
    else:
        device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
        list_ids = list(range(n_gpu_use))
    return device, list_ids


def requires_grad(model, requires=False):
    for param in model.parameters():
        param.requires_grad = requires

def ten2img(tensor):
    tensor.clip_(-1, 1)
    transform = v2.Compose([
        v2.ToPILImage()
    ])
    return transform((tensor + 1) / 2)

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        # if self.writer is not None:
        #     self.writer.add_scalar(key, value)
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.loc[key, "total"] / self._data.loc[key, "counts"]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
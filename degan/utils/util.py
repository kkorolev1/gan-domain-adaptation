import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path
from datetime import datetime

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


def prepare_device(gpus, logger, find_best=False):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu_use = len(gpus)
    n_gpu = torch.cuda.device_count()
    logger.info(f"Num GPUs query: {n_gpu_use}")
    logger.info(f"Num GPUs available: {n_gpu}")
    if n_gpu_use > 0 and n_gpu == 0:
        logger.info(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
        gpus = []
    if n_gpu_use > n_gpu:
        logger.info(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    if n_gpu_use == 1:
        id = find_device() if find_best else gpus[0]
        device = torch.device(f"cuda:{id}")
    else:
        device = torch.device("cuda" if n_gpu_use > 0 else "cpu")
    return device, gpus


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

def get_tril_mask(size):
    mask = torch.ones((size, size), dtype=torch.bool)
    return torch.tril(mask).logical_not()

def nan_hook(self, inp, output):
    if isinstance(output, torch.Tensor):
        output = [output]
    for i, out in enumerate(output):
        if not isinstance(out, torch.Tensor):
            continue
        nan_mask = torch.isnan(out)
        if nan_mask.any():
            print("In", self.__class__.__name__)
            raise RuntimeError(f"Found NAN in output {i} at indices: ", nan_mask.nonzero(), "where:", out[nan_mask.nonzero()[:, 0].unique(sorted=True)])

def setup_checkpoint_dir(config, run_id=None):
    save_dir = Path(config["trainer"]["save_dir"])
    
    exper_name = config["name"]
    if run_id is None:  # use timestamp as default run-id
        run_id = datetime.now().strftime(r"%m%d_%H%M%S")
    checkpoint_dir = save_dir / "models" / exper_name / run_id

    # make directory for saving checkpoints and log.
    exist_ok = run_id == ""
    checkpoint_dir.mkdir(parents=True, exist_ok=exist_ok)

    return checkpoint_dir

class MetricTracker:
    def __init__(self, *keys):
        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"])
        self.reset()

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.loc[key, "total"] += value * n
        self._data.loc[key, "counts"] += n
        self._data.loc[key, "average"] = self._data.loc[key, "total"] / self._data.loc[key, "counts"]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)

    def keys(self):
        return self._data.total.keys()
    
class FadeInScheduler:
    def __init__(self, start=0.1, end=1.0, annealing_steps=1000):
        self.start = start
        self.end = end
        self.schedule = torch.linspace(start, end, annealing_steps)
        self.step = 0
    
    def item(self):
        if self.step >= len(self.schedule):
            return self.end
        return self.schedule[self.step]
    
    def update(self):
        self.step += 1

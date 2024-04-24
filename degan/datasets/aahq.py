import torch
from degan.base import BaseDataset

class AAHQDataset(BaseDataset):
    def __init__(self, root_path, limit=None, transform=None):
        super().__init__(root_path, limit, transform)


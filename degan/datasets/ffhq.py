from glob import glob
import os

from degan.base import BaseDataset


class FFHQDataset(BaseDataset):
    def __init__(self, root_path, limit=None, transform=None):
        super().__init__(root_path, limit, transform)
    def _find_paths(self, root_path):
        return list(glob(os.path.join(root_path, "**/*.png"), recursive=True)) 

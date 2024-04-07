from glob import glob
import os

from degan.base import BaseDataset


class FFHQDataset(BaseDataset):
    def __init__(self, root_path, size):
        super().__init__(root_path, size)
    def _find_paths(self, root_path):
        return list(glob(os.path.join(root_path, "**/*.png"), recursive=True)) 


# dataset = FFHQDataset("datasets/ffhq-dataset/images1024x1024")
# print(len(dataset))
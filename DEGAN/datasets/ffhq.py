from glob import glob
import os

from DEGAN.base import BaseDataset


class FFHQDataset(BaseDataset):
    def __init__(self, root_path):
        super().__init__(root_path)
    def _find_paths(self, root_path):
        return list(glob(os.path.join(root_path, "*/*.png"))) 


# dataset = FFHQDataset("datasets/ffhq-dataset/images1024x1024")
# print(len(dataset))
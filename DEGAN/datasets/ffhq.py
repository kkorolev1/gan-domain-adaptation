import torch
from glob import glob
import os
from pathlib import Path

class FFHQDataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        super().__init__()
        assert os.path.exists(root_path), "Root path to FFHQ doesn't exist"
        self.root_path = Path(root_path)
        self.paths = list(glob(os.path.join(self.root_path, "*/*.png")))

    def __getitem__(self, index):
        raise NotImplementedError()
    
    def __len__(self):
        return len(self.paths)
    
    def get_collate(self):
        return None
    
dataset = FFHQDataset("datasets/ffhq-dataset/images1024x1024")
import torch
from glob import glob
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2
from random import shuffle, seed

class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_path, limit=None, transform=None):
        super().__init__()
        assert os.path.exists(root_path), "Root path to dataset doesn't exist"
        self.root_path = Path(root_path)
        self.paths = self._find_paths(self.root_path)
        if limit is not None:
            seed(1000)
            shuffle(self.paths)
            self.paths = self.paths[:limit]
        self.transform = transform

    def _find_paths(self, root_path):
        return list(glob(os.path.join(root_path, "*.png")))

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.paths)
    
    def get_collate(self):
        def collate_fn(batch):
            batch_dict = {"domain_img": torch.cat([x.unsqueeze(0) for x in batch], dim=0)}
            return batch_dict
        return collate_fn
import torch
from glob import glob
import os
from pathlib import Path
from PIL import Image
from torchvision.transforms import v2


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, root_path):
        super().__init__()
        assert os.path.exists(root_path), "Root path to dataset doesn't exist"
        self.root_path = Path(root_path)
        self.paths = self._find_paths(self.root_path)
        self.transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def _find_paths(self, root_path):
        return list(glob(os.path.join(root_path, "*.png")))

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.transform(image)
    
    def __len__(self):
        return len(self.paths)
    
    def get_collate(self):
        def collate_fn(batch):
            batch_dict = {"domain_img": torch.cat([x.unsqueeze(0) for x in batch], dim=0)}
            return batch_dict
        return collate_fn
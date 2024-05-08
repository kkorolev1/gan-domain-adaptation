import torch
import os
from pathlib import Path
from glob import glob
from PIL import Image
from random import shuffle, seed, choice


class DomainLatentDataset(torch.utils.data.Dataset):
    SEED = 1000

    def __init__(self, root_path, limit=None, domain_limit=None, latent_limit=None, merge_all=False, transform=None, sample_latent=False):
        super().__init__()
        assert os.path.exists(root_path), "Root path to dataset doesn't exist"
        seed(DomainLatentDataset.SEED)

        self.root_path = Path(root_path)
        
        domains_paths = self._truncate_paths(self._find_domains(self.root_path), domain_limit)
        latents_paths = self._truncate_paths(self._find_latents(self.root_path), latent_limit)
        paths = []
        
        if merge_all:
            for domain_path in domains_paths:
                for latent_path in latents_paths:
                    paths.append(self._create_path_mapping(domain_path, latent_path, sample_latent))
        else:
            for domain_path in domains_paths:
                latent_path = choice(latents_paths) if len(latents_paths) > 0 else None
                paths.append(self._create_path_mapping(domain_path, latent_path, sample_latent))
    
        self.paths = self._truncate_paths(paths, limit)
        self.transform = transform
        self.sample_latent = sample_latent
    
    def _create_path_mapping(self, domain_path, latent_path, sample_latent=False):
        mapping = {
            "domain_path": domain_path
        }
        if not sample_latent:
            mapping["latent_path"] = latent_path
        return mapping

    def _truncate_paths(self, paths, limit):
        if limit is None:
            return paths
        shuffle(paths)
        return paths[:limit]

    def _find_domains(self, root_path):
        return list(glob(str(root_path / "domains" / "*"), recursive=True))

    def _find_latents(self, root_path):
        return list(glob(str(root_path / "latents" / "*"), recursive=True))
        
    def __getitem__(self, index):
        path_dict = self.paths[index]
        image = Image.open(path_dict["domain_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.sample_latent:
            latent = torch.randn(512)
        else:
            latent = torch.load(path_dict["latent_path"])
        return {"domain_img": image, "latent": latent, "domain_path": path_dict["domain_path"]}
    
    def __len__(self):
        return len(self.paths)
    
    def get_collate(self):
        def collate_fn(batch):
            batch_dict = {
                "domain_img": torch.cat([x["domain_img"].unsqueeze(0) for x in batch], dim=0),
                "latent": torch.cat([x["latent"].unsqueeze(0) for x in batch], dim=0),
                "domain_path": [x["domain_path"] for x in batch]
            }
            return batch_dict
        return collate_fn
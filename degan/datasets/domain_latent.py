import torch
import os
from pathlib import Path
from glob import glob
from PIL import Image
from random import shuffle, seed
from itertools import chain


class DomainLatentDataset(torch.utils.data.Dataset):
    SEED = 1000

    def __init__(self, root_path, limit=None, domain_limit=None, latent_limit=None, transform=None, sample_latent=False, batch_expand_mult=1, mean_emb_path=None):
        super().__init__()
        assert os.path.exists(root_path), "Root path to dataset doesn't exist"

        self.root_path = Path(root_path)
        
        domains_paths = self._truncate_paths(self._find_domains(self.root_path), domain_limit)
        latents_paths = self._truncate_paths(self._find_latents(self.root_path), latent_limit)
        paths = []
        
        if not sample_latent:
            for domain_path in domains_paths:
                for latent_path in latents_paths:
                    paths.append(self._create_path_mapping(domain_path, latent_path, mean_emb_path, sample_latent))
        else:
            for domain_path in domains_paths:
                paths.append(self._create_path_mapping(domain_path, None, mean_emb_path, sample_latent))
    
        self.paths = self._truncate_paths(paths, limit)
        self.transform = transform
        self.sample_latent = sample_latent
        self.batch_expand_mult = batch_expand_mult
        self.mean_emb_path = mean_emb_path
    
    def _create_path_mapping(self, domain_path, latent_path, mean_emb_path=None, sample_latent=False):
        mapping = {
            "domain_path": domain_path,
            "domain_emb_path": self.root_path / "domains_clip" / domain_path.name
        }
        if mean_emb_path is not None:
            mapping["src_proj_emb_path"] = mean_emb_path
        else:
            mapping["src_proj_emb_path"] = self.root_path / "projections_clip" / domain_path.name
        if not sample_latent:
            mapping["latent_path"] = latent_path
        return mapping

    def _truncate_paths(self, paths, limit):
        if limit is None:
            return paths
        seed(DomainLatentDataset.SEED)
        shuffle(paths)
        return paths[:limit]

    def _find_domains(self, root_path):
        return [Path(path) for path in glob(str(root_path / "domains" / "*"), recursive=True)]

    def _find_latents(self, root_path):
        return [Path(path) for path in glob(str(root_path / "latents" / "*"), recursive=True)]
        
    def __getitem__(self, index):
        path_dict = self.paths[index]
        domain_img = Image.open(path_dict["domain_path"]).convert("RGB")
        if self.transform is not None:
            domain_img = self.transform(domain_img)
        item_dict = {
            "domain_img": domain_img,
            "domain_path": path_dict["domain_path"].name,
            "domain_emb": torch.load(path_dict["domain_emb_path"], map_location="cpu"),
            "src_proj_emb": torch.load(path_dict["src_proj_emb_path"], map_location="cpu")
        }
        if not self.sample_latent:
            item_dict["latent"] = torch.load(path_dict["latent_path"], map_location="cpu")
        return item_dict
    
    def __len__(self):
        return len(self.paths)
    
    def get_collate(self):
        def collate_fn(batch):
            batch_dict = {
                "domain_img": torch.cat([x["domain_img"].repeat(self.batch_expand_mult, 1, 1, 1) for x in batch], dim=0),
                "domain_emb": torch.cat([x["domain_emb"].repeat(self.batch_expand_mult, 1) for x in batch], dim=0),
                "src_proj_emb": torch.cat([x["src_proj_emb"].repeat(self.batch_expand_mult, 1) for x in batch], dim=0),
                "domain_path": list(chain(*[[x["domain_path"]] * self.batch_expand_mult for x in batch])),
                "batch_expand_mult": self.batch_expand_mult
            }
            if self.sample_latent:
                batch_dict["latent"] = torch.randn(len(batch) * self.batch_expand_mult, 512)
            else:
                batch_dict["latent"] = torch.cat([x["latent"].repeat(self.batch_expand_mult, 1) for x in batch], dim=0)
            return batch_dict
        return collate_fn
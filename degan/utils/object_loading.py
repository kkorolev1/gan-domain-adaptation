from operator import xor

from torch.utils.data import ConcatDataset, DataLoader

from hydra.utils import instantiate


def get_dataloaders(config):
    dataloaders = {}
    for split, params in config["data"].items():
        num_workers = params.get("num_workers", 1)

        if split == 'train':
            drop_last = True
        else:
            drop_last = False

        # create and join datasets
        datasets = []
        for ds in params["datasets"]:
            datasets.append(instantiate(ds))
        assert len(datasets)
        if len(datasets) > 1:
            dataset = ConcatDataset(datasets)
        else:
            dataset = datasets[0]

        assert "batch_size" in params, \
            "You must provide batch_size for each split"
        if "batch_size" in params:
            bs = params["batch_size"]
            shuffle = True
        else:
            raise Exception()

        # Fun fact. An hour of debugging was wasted to write this line
        assert bs <= len(dataset), \
            f"Batch size ({bs}) shouldn't be larger than dataset length ({len(dataset)})"

        # create dataloader
        dataloader = DataLoader(
            dataset, batch_size=bs, collate_fn=dataset.get_collate(),
            shuffle=shuffle, num_workers=num_workers,
            drop_last=drop_last
        )
        dataloaders[split] = dataloader
    return dataloaders
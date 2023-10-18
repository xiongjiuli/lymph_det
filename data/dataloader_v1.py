"""Module containing dataloader related functionality."""

import torch
from torch.utils.data import DataLoader

from data.dataset_v1 import detDataset



def get_loader(config, split, batch_size=None):
    if not batch_size:
        batch_size = config['batch_size']

    # Init collator
    collator = detCollator(config)
    shuffle = False if split in ['testing', 'validation'] else config['shuffle']

    dataset = detDataset(config, split)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=config['num_workers'], collate_fn=collator
    )
    return dataloader


class detCollator:
    def __init__(self, config):
        self._bbox_padding = config['bbox_padding']

    def __call__(self, batch):
        batch_images = []
        batch_hmap = []
        batch_mask = []
        batch_offset = []
        batch_whd = []
        batch_name = []
        for dct in batch:
            image = dct['input'] 
            hmap = dct['hmap']
            offset = dct['offset']
            mask = dct['mask']
            whd = dct['whd']
            name = dct['name']
    
            batch_images.append(image)
            batch_hmap.append(hmap)
            batch_mask.append(mask)
            batch_offset.append(offset)
            batch_whd.append(whd)
            batch_name.append(name)

        batch_images = [torch.from_numpy(np_array) for np_array in batch_images]

        dct = {}
        dct['hmap'] = torch.stack(batch_hmap).unsqueeze(1)
        dct['input'] = torch.stack(batch_images).unsqueeze(1)
        dct['mask'] = torch.stack(batch_mask)
        dct['offset'] = torch.stack(batch_offset)
        dct['whd'] = torch.stack(batch_whd)
        dct['name'] = batch_name
        return dct

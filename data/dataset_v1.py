"""Module containing the dataset related functionality."""

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchio 
import sys
import os
from IPython import embed
import torch.utils.data as data
from time import time 
import random
import csv
from preprocessor import generate_data, crop_data_region, region_generate
from data.cascade import stage_one

def read_names_from_csv(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        names = []
        for row in reader:
            # print(row)
            name = row[0]
            names.append(name)
    return names


class detDataset(Dataset):
    """Dataset class of the det project."""
    def __init__(self, config, split):
        assert split in ['training', 'validation', 'testing']
        self._config = config
        data_dir = self._config['lymph_nodes_data_path']
        csv_file_path = os.path.join(self._config['csv_names_root_path'], f'{split}_names.csv')
        self._names = read_names_from_csv(csv_file_path)
        self._split = split

    def __len__(self):
        return len(self._names)

    def __getitem__(self, idx):
        if self._config['overfit']:
            idx = 0

        if self._split == 'training':
            random.seed(1)
            random.shuffle(self._names)

        name = self._names[idx]
        data_root_path = Path(self._config['lymph_nodes_data_path'])
        image_path = data_root_path.joinpath(f'{self._split}_npy').joinpath(f'{name}_image.npy')
        hmap_path = data_root_path.joinpath(f'{self._split}_npy').joinpath(f'{name}_hmap.npy')
        region_path = data_root_path.joinpath(f'{self._split}_npy').joinpath(f'{name}_region.npy')

        if image_path.exists() and hmap_path.exists():
            data = np.load(image_path)
            hmap = np.load(hmap_path)
        else:
            data, hmap = generate_data(data_root_path, self._split, name)

        if region_path.exists():
            region = np.load(region_path)
        else:
            region = region_generate(data_root_path, self._split, name)

        if self._config['cascade']:
            if random.random() > 0.5:
                center = stage_one(self._config, name, self._split)
            else:
                non_zero_positions = np.argwhere(region != 0)
                center = random.choice(non_zero_positions)
        else:
            non_zero_positions = np.argwhere(region != 0)
            center = random.choice(non_zero_positions)

        dct = crop_data_region(self._split, name, data, hmap, center, self._config['crop_size'])

        return dct

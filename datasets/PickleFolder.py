import torch
import pickle
import random

import numpy as np
import utils.boxcox as boxcox

from pathlib import Path
from torch.utils.data import Dataset


# Random number generators
def gen_valid_ratio_uniform_random(vmin=0.0003, vmax=0.0025):
    
    def valid_ratio_uniform_random(vmin=vmin, vmax=vmax):
        return random.uniform(vmin, vmax)
        
    return valid_ratio_uniform_random


def gen_valid_ratio_biased_random(scale=0.3, vmin=0.0001, vmax=0.9):
    
    def valid_ratio_biased_random(scale=scale, vmin=vmin, vmax=vmax):
        x = np.random.exponential(scale=scale)
        x /= (x + 1)
        return vmin + x * (vmax - vmin)
        
    return valid_ratio_biased_random
    

class PickleFolderDataset(Dataset):
    def __init__(
            self,
            folder,
            n_repeats_with_random_splits=1,
            random_func=gen_valid_ratio_uniform_random(),
            ):

        super().__init__()
        self.n_repeats_with_random_splits = n_repeats_with_random_splits
        self.random_func = random_func

        folder = Path(folder)
        self.data_files = list(folder.glob("*.pkl"))
        self.data_files.sort()

        with open(folder / 'times', 'rb') as f:
            self.times = pickle.load(f)

        self.data_files = np.array(self.data_files)

        assert len(self.times) == len(self.data_files)

    def subset_times_index(self, times_index):
        self.times = self.times[times_index]
        self.data_files = self.data_files[times_index]

    def __len__(self):
        return len(self.times) * self.n_repeats_with_random_splits

    def sample(self, sample_idx, valid_ratio, include_time=True, return_original=False):
        time_idx = sample_idx // self.n_repeats_with_random_splits

        # Load the field at a given time
        with open(Path(self.data_files[time_idx]), 'rb') as f:
            full_field = pickle.load(f).astype(np.float32)  # (H, W)

        if return_original:
            return full_field

        full_field = boxcox.transform(full_field)
            
        # Generate a random mask
        valid_mask = (np.random.rand(*full_field.shape) < valid_ratio).astype(np.float32)
        valid_masked_field = full_field * valid_mask

        # Convert to torch tensors with channel dimension
        full_field = torch.from_numpy(full_field).unsqueeze(0)                   # (1, H, W)
        valid_masked_field = torch.from_numpy(valid_masked_field).unsqueeze(0)   # (1, H, W)
        valid_mask = torch.from_numpy(valid_mask).unsqueeze(0)                   # (1, H, W)

        sample = {
            'full_field': full_field,
            'masked_field': valid_masked_field,
            'null_mask': 1. - valid_mask,         # 0 for known regions and 1 for masked regions
        }

        if include_time:
            sample['time'] = self.times[time_idx]
            
        return sample

    def __getitem__(self, sample_idx):
        valid_ratio = self.random_func()
        return self.sample(sample_idx, valid_ratio, include_time=False)

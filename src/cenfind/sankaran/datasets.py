import contextlib
import os

import numpy as np
import torch
from spotipy.utils import points_to_prob, normalize_fast2d
from torch.utils.data import Dataset

from cenfind.core.data import Dataset as Cenfind_dataset, Field
from cenfind.sankaran.helpers import (preprocess_image, compute_weight, compute_labelmap, compute_mindist_map)


class FociDatasetSankaran(torch.utils.data.Dataset):
    def __init__(self, dataset: Cenfind_dataset, split=None, channel=None):
        self.dataset = dataset
        self.pairs = self.dataset.pairs(split)
        self.channel = channel

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        field_path, random_channel = self.pairs[idx]
        if self.channel is None:
            self.channel = random_channel
        field = Field(field_path, self.dataset)
        data = field.projection
        data = data[1:4, :, :]
        data = data.astype(float)
        data = preprocess_image(data)

        positions = field.annotation(self.channel)

        mask = np.zeros_like(data[0, :, :])

        min_dist_map = compute_mindist_map(mask, positions)
        labelmap = compute_labelmap(mask, min_dist_map)
        weight = compute_weight(labelmap, min_dist_map)

        return data, labelmap, weight


class FociDataset(Dataset):
    def __init__(self, dataset: Cenfind_dataset, transform=None):
        self.dataset = dataset
        self.pairs = self.dataset.pairs()
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        field_path, channel = self.pairs[idx]
        field = Field(field_path, self.dataset)
        data = field.channel(channel)
        positions = field.annotation(channel)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            image = normalize_fast2d(data)
        mask = points_to_prob(positions[:, [1, 0]],  # because it works with x, y
                              shape=image.shape,
                              sigma=1)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        return image, mask

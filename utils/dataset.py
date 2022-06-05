"""
Helper to load the 102 Category Flower Dataset.
"""

import os

import numpy as np
import scipy.io
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F


class FlowerDataset(Dataset):

    def __init__(self, data_path: str, img_size: int = 64):
        self.data_path = data_path
        self.img_size = img_size
        self.img_map = {}
        self.img_cache = {}
        self.labels = torch.tensor(
            scipy.io.loadmat(f'{data_path}/imagelabels.mat')['labels'][0], dtype=torch.long
        ) - 1
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._map_images()

    def _map_images(self):
        files = sorted(os.listdir(f'{self.data_path}/jpg'))
        files = [f for f in files if f.endswith('.jpg')]
        for i, f in enumerate(files):
            self.img_map[i] = (f, self.labels[i])

    def __len__(self):
        return len(self.img_map)

    def __getitem__(self, idx):
        if idx in self.img_cache:
            return self.img_cache[idx]
        img_name, label = self.img_map[idx]
        img = Image.open(f'{self.data_path}/jpg/{img_name}')
        img = self._transform_img(img).to(self.device)
        label = label.to(self.device)
        self.img_cache[idx] = img, label
        return img, label

    def _transform_img(self, img):
        img = F.to_tensor(np.array(img))
        img = F.resize(img, [self.img_size, self.img_size])
        # range between -1 and 1 to match Tanh output activation
        img = F.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.51, 0.51, 0.51])
        return img
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from sklearn import preprocessing
import os


class BalloonDataset(Dataset):
    def __init__(self, transform=None):
        cur_path = os.path.dirname(__file__)
        self.path = os.path.join(cur_path, 'balloon/datasets.csv')
        #self.path = '/Users/mojskarb/stardust/data/balloon/datasets.csv'
        self.transform = transform

        xy = np.loadtxt(self.path, delimiter=',', dtype=np.float32)
        self.lats = xy[:, 0]
        self.lons = xy[:, 1]
        self.alts = xy[:, 2]

        self.x = xy[:, 3:-3]
        
        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(xy[:, -3:])

        self.n_samples = xy.shape[0]

    def __getitem__(self, index):
        if self.transform:
            self.x = self.transform(self.x)
        return self.x[index], self.y[index]

    def __len__(self):
        return self.n_samples
    
    def get_lat(self, index):
        return self.lats[index]

    def get_lon(self, index):
        return self.lons[index]
    
    def get_alt(self, index):
        return self.alts[index]



import os
import os.path as osp
import pickle

import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset):
    def __init__(self, data_path, mode):
        super(BaseDataset, self).__init__()

        self.image_data = self._load(
            osp.join(data_path, mode + "_imgs.pickle")
        )
        self.label_data = self._load(
            osp.join(data_path, mode + "_labels.pickle")
        )

        self.image_size = self.image_data.shape[:2]
        self.len = self.image_data.shape[2]

        print("-" * 25, mode, "-" * 25)
        print("Image shape: {}".format(self.image_size))
        print("Total images: {}".format(self.len))
        print("-" * 50)

    def __len__(self):
        return self.len

    @staticmethod
    def _load(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
        return data

    def __getitem__(self, item):
        image = self.image_data[:, :, item].astype(np.float32)
        label = self.label_data[:, :, item].astype(np.float32)

        # transform

        data_batch = {"image": image, "label": label}
        return data_batch

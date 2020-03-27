import os
import os.path as osp

import torch
from torch.utils.data import Dataset

import nibabel


class BaseDataset(Dataset):
    def __init__(self, data_path):
        super(BaseDataset, self).__init__()

        self.image_path = osp.join(data_path, "imgs")
        self.image_list = sorted(os.listdir(self.image_path))
        self.label_path = osp.join(data_path, "labels")
        self.label_list = sorted(os.listdir(self.label_path))

        self.num_images = len(self.image_path)

        # load sample image

    def __len__(self):
        pass

    @staticmethod
    def _read(file):
        instance = nibabel.load(file)
        data = instance.get_fdata()

        return data

    def __getitem__(self, item):
        image = self._read(
            osp.join(self.image_path, self.image_list[item])
        )
        label = self._read(
            osp.join(self.label_path, self.label_list[item])
        )

        return image, label

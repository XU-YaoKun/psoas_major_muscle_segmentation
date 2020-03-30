import os
import os.path as osp

import torch
from torch.utils.data import Dataset
import numpy as np

import nibabel


class BaseDataset(Dataset):
    def __init__(self, data_path):
        super(BaseDataset, self).__init__()

        self.image_path = osp.join(data_path, "imgs")
        self.image_list = sorted(os.listdir(self.image_path))
        self.label_path = osp.join(data_path, "labels")
        self.label_list = sorted(os.listdir(self.label_path))

        self.num_images = len(self.image_list)

        # load sample image
        sample_img = self._read(
            osp.join(self.image_path, self.image_list[-1])
        )
        self.img_shape = sample_img.shape
        self.z = self.img_shape[-1]

        print("Total images: {}".format(self.num_images))
        print("Image shape: {}".format(self.img_shape))
        print(
            "Total training samples: [{} x {} = {}]".format(
                self.num_images,
                self.z,
                self.num_images * self.z,
            )
        )

    def __len__(self):
        return self.num_images * self.z

    @staticmethod
    def _read(file):
        instance = nibabel.load(file)
        data = instance.get_fdata()
        return data

    def __getitem__(self, item):
        image_idx = int(item / self.z)
        z_idx = int(item % self.z)

        _image = self._read(
            osp.join(
                self.image_path, self.image_list[image_idx]
            )
        )
        _label = self._read(
            osp.join(
                self.label_path, self.label_list[image_idx]
            )
        )

        image = _image[:, :, z_idx].astype(np.float32)
        label = _label[:, :, z_idx].astype(np.float32)

        data_batch = {"image": image, "label": label}
        return data_batch

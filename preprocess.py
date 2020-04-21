import os
import os.path as osp
import pickle

import nibabel
import numpy as np


def _read(file):
    instance = nibabel.load(file)
    data = instance.get_fdata()
    return data


def _dump(data_path, name):
    file_list = sorted(os.listdir(data_path))

    data = np.empty([512, 512, 0], dtype=np.float32)
    for file in file_list:
        print("read {}...".format(osp.join(data_path, file)))
        imgs = _read(osp.join(data_path, file))
        data = np.concatenate([data, imgs], axis=2)

    dump_path = osp.join("data", name)
    with open(dump_path, "wb") as f:
        pickle.dump(data, f)

    print("Dump {} to {}".format(data_path, dump_path))
    return


def _dump_dataset(mode):
    _dump(osp.join("data", mode, "imgs"), mode + "_imgs.pickle")
    _dump(
        osp.join("data", mode, "labels"),
        mode + "_labels.pickle",
    )


def main():
    _dump_dataset("train")
    _dump_dataset("test")


if __name__ == "__main__":
    main()

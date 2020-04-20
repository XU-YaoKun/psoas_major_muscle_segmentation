from torch.utils.data import DataLoader, random_split

from .dataset import BaseDataset


def build_dataloader(cfg):
    trainset = BaseDataset(cfg.DATA_PATH, mode="train")
    testset = BaseDataset(cfg.DATA_PATH, mode="test")

    trainloader = DataLoader(
        trainset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=True,
    )

    testloader = DataLoader(
        testset,
        batch_size=cfg.BATCH_SIZE,
        num_workers=cfg.NUM_WORKERS,
        shuffle=False,
    )

    return trainloader, testloader

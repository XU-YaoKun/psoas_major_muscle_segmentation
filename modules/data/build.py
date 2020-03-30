from torch.utils.data import DataLoader, random_split

from .dataset import BaseDataset


def build_dataloader(cfg):
    dataset = BaseDataset(cfg.DATA_PATH)
    n_test = int(len(dataset) * cfg.TEST_PERCENT)
    n_train = len(dataset) - n_test

    trainset, testset = random_split(dataset, [n_train, n_test])

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

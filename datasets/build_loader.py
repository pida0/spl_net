'''
Author: sy
Date: 2021-04-01 11:39:11
LastEditors: your name
LastEditTime: 2021-04-25 18:52:53
Description: file content
'''
from .celeba import CelebA
from .lfwa import LFWA
from .maad import MAAD
import torch

__factory = {
    'CelebA': CelebA,
    'LFWA': LFWA,
    'MAAD': MAAD
    
}


def get_names():
    return __factory.keys()


def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown datasets: {}".format(name))
    return __factory[name](*args, **kwargs)

def make_dataloader(cfg, group=[12, 13, 6, 9]):

    trainset = init_dataset(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='0', type=cfg.DATA.TYPE + '-train', group=group)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=True, num_workers=2)

    valset = init_dataset(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='1', type=cfg.DATA.TYPE + '-val', group=group)
    valloader = torch.utils.data.DataLoader(valset, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=True, num_workers=2)

    testset = init_dataset(cfg.DATA.DATASET_NAME, cfg=cfg.DATA, partition='2', type=cfg.DATA.TYPE + '-val', group=group)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.TRAIN.BATCHSIZE, shuffle=True, num_workers=2)

    return trainloader, valloader, testloader
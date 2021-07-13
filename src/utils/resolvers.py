from bisect import bisect_right
from collections import Counter

import torchvision
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import _LRScheduler

from src.datasets.definitions import SPLIT_TRAIN, SPLIT_VALID
from src.datasets.voc_sbd import DatasetVocSbd
from src.models.model_net_deeplabv3p import ModelNetDeepLabV3Plus
from src.models.model_net_resnet_cifar10 import ModelNetResnetCifar10
from src.models.model_net_lenet5 import ModelNetLenet5


def resolve_imgcls_model(name):
    return {
        'resnet_cifar10': ModelNetResnetCifar10,
        'lenet5': ModelNetLenet5,
    }[name]


def resolve_optimizer(name):
    return {
        'sgd': SGD,
        'adam': Adam,
    }[name]


class MultiStepWarmupLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, num_warmup_steps=0, last_epoch=-1, verbose=False):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        dampen = max(0.0, min(1.0, self.last_epoch / self.num_warmup_steps))
        milestones = list(sorted(self.milestones.elements()))
        return [dampen * base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
                for base_lr in self.base_lrs]


class PolyWarmupLR(_LRScheduler):
    def __init__(self, optimizer, power, num_steps, num_warmup_steps=0, last_epoch=-1, verbose=False):
        self.power = power
        self.num_steps = num_steps
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        dampen = max(0.0, min(1.0, self.last_epoch / self.num_warmup_steps))
        return [base_lr * dampen * (1.0 - min(self.last_epoch, self.num_steps-1) / self.num_steps) ** self.power
                for base_lr in self.base_lrs]


def resolve_imgcls_dataset(cfg):
    if cfg.dataset == 'mnist':
        dataset_train = torchvision.datasets.MNIST(
            cfg.root_datasets[cfg.dataset], train=True, download=cfg.dataset_download,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (1.0,))
            ])
        )
        dataset_valid = torchvision.datasets.MNIST(
            cfg.root_datasets[cfg.dataset], train=False, download=cfg.dataset_download,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (1.0,))
            ])
        )
        return dataset_train, dataset_valid, 10
    transform_train, transform_valid = [], []
    if cfg.dataset in ('cifar10', 'cifar100'):
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = {
            'cifar10': 10,
            'cifar100': 100,
        }[cfg.dataset]
        transform_train += [
            torchvision.transforms.Pad(4),
            torchvision.transforms.RandomResizedCrop(32),
        ]
    else:
        raise NotImplementedError(f'Dataset {cfg.dataset} functionality not implemented')

    transform_train += [
        torchvision.transforms.RandomHorizontalFlip()
    ]
    transform_epilogue = [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ]
    transform_train += transform_epilogue
    transform_valid += transform_epilogue

    transform_train = torchvision.transforms.Compose(transform_train)
    transform_valid = torchvision.transforms.Compose(transform_valid)

    if cfg.dataset in ('cifar10', 'cifar100'):
        dataset_class = {
            'cifar10': torchvision.datasets.CIFAR10,
            'cifar100': torchvision.datasets.CIFAR100,
        }[cfg.dataset]
        dataset_train = dataset_class(
            cfg.root_datasets[cfg.dataset], train=True, transform=transform_train, download=cfg.dataset_download
        )
        dataset_valid = dataset_class(
            cfg.root_datasets[cfg.dataset], train=False, transform=transform_valid, download=cfg.dataset_download
        )
    else:
        raise NotImplementedError(f'Dataset {cfg.dataset} functionality not implemented')

    return dataset_train, dataset_valid, num_classes


def resolve_semseg_dataset(cfg):
    if cfg.dataset == 'voc_sbd':
        dataset_train = DatasetVocSbd(cfg.root_datasets['voc_sbd'], SPLIT_TRAIN, cfg.dataset_download)
        dataset_valid = DatasetVocSbd(cfg.root_datasets['voc_sbd'], SPLIT_VALID, cfg.dataset_download)
    else:
        raise NotImplementedError(f'Dataset {cfg.dataset} functionality not implemented')
    return dataset_train, dataset_valid


def resolve_semseg_model(cfg, num_classes):
    if cfg.model_name == 'deeplabv3p':
        return ModelNetDeepLabV3Plus(cfg, num_classes)
    else:
        raise NotImplementedError(f'Dataset {cfg.dataset} functionality not implemented')

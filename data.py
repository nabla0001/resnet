import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def cifar10_data_loaders(batch_size: int = 128, data_dir: str = 'data') -> tuple[DataLoader, DataLoader, DataLoader]:
    """Returns CIFAR10 train/val/test data loaders

    train:  45k - data augmentation
    val:    5k - no data augmentation
    test:   10k - no data augmentation

    all datasets are normalized
    (1) pixels into [0,1]
    (2) mean/std normalised channel-wise
    """
    channel_mean = (0.4914, 0.4822, 0.4465)
    channel_std = (0.247, 0.243, 0.261)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(channel_mean, channel_std),
        transforms.Pad(4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32)])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(channel_mean, channel_std)])

    # 45k/5k train/val split
    train = torchvision.datasets.CIFAR10(root=data_dir,
                                         train=True,
                                         transform=train_transform,
                                         download=True)
    val = torchvision.datasets.CIFAR10(root=data_dir,
                                         train=True,
                                         transform=test_transform,
                                         download=True)

    idx = torch.randperm(len(train))
    train_idx, val_idx = idx[:45000], idx[45000:]

    train = torch.utils.data.Subset(train, train_idx)
    val = torch.utils.data.Subset(val, val_idx)

    test = torchvision.datasets.CIFAR10(root=data_dir,
                                        train=False,
                                        transform=test_transform,
                                        download=True)

    train_loader = torch.utils.data.DataLoader(train, batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
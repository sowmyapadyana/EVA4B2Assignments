from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class TrainSetLoader():
    def __init__(self, kwargs, batch_size):
        self.kwargs = kwargs
        self.batch_size = batch_size

    def data_loader(self, data_type):
        torch.manual_seed(1)
        if data_type == "train":
            data_load = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                                transform=transforms.Compose([
                                    transforms.RandomRotation((-0.7,0.7), fill=(1,)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
                batch_size=self.batch_size, shuffle=True, **self.kwargs)
        else:
            data_load = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False, transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))
                                ])),
                batch_size=self.batch_size, shuffle=True, **self.kwargs)

        return data_load


    def data_loader_cifar(self, data_type):
        torch.manual_seed(1)

        # transform = transforms.Compose(
        #     [transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Normalize the test set same as training set without augmentation
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


        if data_type == "train":
            data_load = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=True, download=True,
                                transform=transform_train),
                batch_size=self.batch_size, shuffle=True, **self.kwargs)
        else:
            data_load = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data', train=False, transform=transform_test),
                batch_size=self.batch_size, shuffle=True, **self.kwargs)

        return data_load
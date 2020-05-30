from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import albumentations as A
from classalbumentations import AlbumentationImageDataset as AlbClass

class TrainSetLoader():
    def __init__(self, kwargs, batch_size):
        self.kwargs = kwargs
        self.batch_size = batch_size

    def data_loader(self, data_type, dataset):
        if data_type == "train":
            transform = transforms.Compose(
                [transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            # alb = AlbClass()
            # transform = alb.transform()
            # transform = A.Compose({
            #     # A.Resize(200, 300),
            #     # A.CenterCrop(100, 100),
            #     # A.RandomCrop(80, 80),
            #     A.HorizontalFlip(p=0.5),
            #     A.Rotate(limit=(-90, 90)),
            #     A.VerticalFlip(p=0.5),
            #     A.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            #     })

            trainset = datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transform)
            dataloader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size,
                                                    shuffle=True, num_workers=2)

        else:
            transform = transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            testset = datasets.CIFAR10(root='./data', train=False,
                                                download=True, transform=transform)
            dataloader = torch.utils.data.DataLoader(testset, batch_size=self.batch_size,
                                                    shuffle=False, num_workers=2)

        return dataloader
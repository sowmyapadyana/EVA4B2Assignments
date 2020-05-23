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

    def data_loader(self, data_type, dataset):
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
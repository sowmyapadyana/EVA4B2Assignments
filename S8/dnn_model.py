from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# Final network from Session 5 assignment
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net5(nn.Module):
    def __init__(self):
        super(Net5, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net6(nn.Module):
    def __init__(self):
        super(Net6, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net7(nn.Module):
    def __init__(self):
        super(Net7, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)


class Net8(nn.Module):
    def __init__(self):
        super(Net8, self).__init__()
        self.drop1 = nn.Dropout(0.1)                        #                       GRF - 1   J - 1
        self.conv1 = nn.Conv2d(1, 16, 3, padding=0)         #input - 28 OUtput - 26  J - 1  GRF - 3
        self.conv2 = nn.Conv2d(16, 10, 3, padding=0)        #input - 26 OUtput - 24  J - 1  GRF - 5 
        self.pool1 = nn.MaxPool2d(2, 2)                     #input - 24 OUtput - 12  J - 2  GRF - 7
        self.bnm2d1 = nn.BatchNorm2d(10) 
        self.conv3 = nn.Conv2d(10, 10, 3, padding=0)        #input - 12 OUtput - 10  J - 2  GRF - 11
        self.conv4 = nn.Conv2d(10, 16, 3, padding=0)        #input - 10 OUtput - 8   J - 2  GRF - 15
        # self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3, padding=0)        #input - 8  OUtput - 6   J - 2  GRF - 19
        self.conv6 = nn.Conv2d(16, 16, 3, padding=1)        #input - 6  OUtput - 6   J - 2  GRF - 23
        # self.conv7 = nn.Conv2d(16, 16, 3, padding=1)      #input - 6  OUtput - 6 
        self.gap = nn.AvgPool2d(kernel_size=6)
        self.conv8 = nn.Conv2d(16,10,1)                                             #J = 2  GRF - 23 

    def forward(self, x):
        x = self.bnm2d1(self.pool1(self.drop1(F.relu(self.conv2(self.drop1(F.relu(self.conv1(x))))))))
        x = self.bnm2d2(self.drop1(F.relu(self.conv4(self.drop1(F.relu(self.conv3(x)))))))
        x = self.conv6(self.drop1(F.relu(self.conv5(x))))
        # x = self.conv7(x)
        x = self.gap(x)
        x = self.conv8(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)
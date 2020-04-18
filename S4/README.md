# Assignment 4

## Analysis

### Dropout layer:
1. Dropout (20%) and Batch normalization used on the given model.
2. Dropout is used to avoid overfitting of the data. Hence dropout layer is used with every convolution layer conv1 to conv4.
3. Since using dropout close to output layer will result in loss of information, convolutions conv5 to conv7 donot have dropout.

### Batch normalization:
1. Batch normalization is used to normalize the values in a particular range, to make the calculation easier.
2. Batch normalization is done twice in th iteration that gave me 99.45% accuracy
3. When Batch normalization is used with every convolution layer, it reduced the accuracy

### Parameters:
1. To have the number of parameters less than 20k, reduced the number of kernels/channels in each layer.
2. I used combination of 16 and 32 channels in my solution.

# Solution that gave 99.45% accuracy:

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.drop5 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1) 
        self.conv2 = nn.Conv2d(32, 16, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.bnm2d1 = nn.BatchNorm2d(16) 
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.bnm2d2 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(16, 16, 3)
        self.conv6 = nn.Conv2d(16, 32, 3)
        self.conv7 = nn.Conv2d(32, 10, 3)

    def forward(self, x):
        x = self.bnm2d1(self.pool1(F.relu(self.drop2(self.conv2(F.relu(self.drop2(self.conv1(x))))))))
        x = self.bnm2d2(self.pool2(F.relu(self.drop2(self.conv4(F.relu(self.drop2(self.conv3(x))))))))
        x = F.relu(self.conv6(F.relu(self.conv5(x))))
        x = self.conv7(x)
        x = x.view(-1, 10)
        return F.log_softmax(x)

Model summary
----------------------------------------------------------------
        Layer (type)               Output Shape         Param 
----------------------------------------------------------------
            Conv2d-1           [-1, 32, 28, 28]             320
           Dropout-2           [-1, 32, 28, 28]               0
            Conv2d-3           [-1, 16, 28, 28]           4,624
           Dropout-4           [-1, 16, 28, 28]               0
         MaxPool2d-5           [-1, 16, 14, 14]               0
       BatchNorm2d-6           [-1, 16, 14, 14]              32
            Conv2d-7           [-1, 16, 14, 14]           2,320
           Dropout-8           [-1, 16, 14, 14]               0
            Conv2d-9           [-1, 16, 14, 14]           2,320
          Dropout-10           [-1, 16, 14, 14]               0
        MaxPool2d-11             [-1, 16, 7, 7]               0
      BatchNorm2d-12             [-1, 16, 7, 7]              32
           Conv2d-13             [-1, 16, 5, 5]           2,320
           Conv2d-14             [-1, 32, 3, 3]           4,640
           Conv2d-15             [-1, 10, 1, 1]           2,890
----------------------------------------------------------------
Total params: 19,498
Trainable params: 19,498
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.74
Params size (MB): 0.07
Estimated Total Size (MB): 0.81
----------------------------------------------------------------



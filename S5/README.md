# Assignment 5

### Attempt 1:
#### Target:
    A basic working skeleton for digit classification using MNIST dataset
    Reduced number of channels in each layer for the training to be faster

#### Result:
    Parameters : 49,450
    Epochs     : 20  
    Train accuracy = 99.60%
    Test accuracy = 99.17%

#### Analysis:
    Since test accuracy is slightly less than train accuracy - there is small amount of overfitting
    The test accuracy is fluctuating between 98% to 99.17%

### Attempt 2:
#### Target:
    Add dropout layers to minimize overfitting. 10% dropout to every convolution layer
    Batch normalization layers to normalize the input data. Added batch normalization after max pooling
    Max epochs = 15

#### Result:
    Parameters : 
      1st run : 49,578
      2nd run : 17,930

#### Analysis:
    Tried dropout = 10% and 20%, yet there is lot of overfitting.
    Reduced the channel size to 16 in multiple layers and kept dropout to 10%. This has solved the overfitting problem. The model accuracy is gradually increasing, yet there is no consistency


### Attempt 3:
#### Target:
    Max epochs = 15
    Number of parameters < 10k
    Adding GAP layer

#### Result:
    1st run:
      Change: Added GAP layer with kernel size = 5
      Parameters : 8,634
      Train accuracy = 98.80%
      Test accuracy = 98.55%

    2nd run : 
      Change : GAP layer with kernel size = 7
               Added extra convolution layer before GAP
      Parameters : 8,634
      Train accuracy = 98.54%
      Test accuracy = 98.28%


#### Analysis:
    1st iteration - Changed the skeleton, removed 3 conv layers and added a GAP layer - Got max accuracy of 98.55%
    2nd iteration - Added a conv layer before GAP, GAP layer with kernel size 7 - got accuracy of 98.28%


### Attempt 4:
#### Target:
    Max epochs = 15
    Number of parameters < 10k
    Adding a layer after GAP layer to increase the capacity

#### Result:
    1st run:
      Change : Added a layer after GAP
      Parameters : 9,674
      Train accuracy : 98.69%
      Test accuracy : 97.59%

    2nd run:
      Change : Removed the second max pool layer, added more conv layers in its place, reduced channels to keep the parameters below 10k
      Parameters : 8,838 
      Train accuracy : 98.85%
      Test accuracy : 99.13%

#### Analysis:
    1st iteration - Accuracy reduced
    2nd iteration - Accuracy improved, no over fitting, the model can be further worked upon to increase the accuracy


### Attempt 5:
#### Target:
    Add Data augmentation to improve accuracy

#### Result:
    1st run:
      Change : Added GAP layer with kernel size = 6
      Parameters : 8,838
      Train accuracy = 98.71%
      Test accuracy = 99.21%%
 
#### Analysis:
    1st iteration - No overfitting, we can improve the model

    
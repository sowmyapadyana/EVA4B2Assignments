# EVA4B2Assignments
Assignments done for the EVA4B2 cource 

### S8 assignment
#### To run the ResNet18 model on CIFAR10 dataset

##### Analysis:
1. Tried with a basic model without any regularization methods. 
    The model trained with accuracy = 10% in every epoch
2. Tried L1 and L2 regularizations separately, but no change in accuracy
3. Tried data augmentation techniques - horizontal flip, normalization, random crop, yet the accuracy didnt improve from 10%. Not able to figure what the problem is.
4. Batch size and learning rate also changed - yet no luck.
5. Used different base code for training the model. Got 83% accuracy this time

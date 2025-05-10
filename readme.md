# Image Classification

### Aim: The aim of this project is to classify the CIFAR10 dataset consisting of 60000 color images into 10 classes using neural networks and deep learning. 

### Basic Architecture:
**DataLoader:** PyTorch’s torchvision.datasets was used to load the CIFAR-10 dataset and apply standard transformations: conversion to tensors and normalization using channel-wise mean and standard deviation. Train and test datasets are loaded separately.
**Intermediate Blocks:** Each intermediate block contains multiple parallel convolutional layers with the same input/output channels. The block is implemented as a custom PyTorch module, accepting lists of kernel sizes, strides, and paddings. Outputs from each branch are combined (e.g., summed or weighted) to form the block’s output.
**Output Block:** The output block applies Global Average Pooling to compress feature maps, followed by a fully connected layer that produces class logits for the 10 CIFAR-10 classes.
**CIFAR10Model:** This class defines the overall model. It starts with an intermediate block mapping 3 → 64 channels, followed by batch normalization and ReLU. Another block maps 64 → 128 channels. After similar normalization and activation, the output block handles the final classification. 
**Evaluate Model:** The evaluation function sets the model to evaluation mode, disables gradients, and computes loss and accuracy on the test dataset.
**Train Model:** The training loop runs for 100 epochs. In each epoch, the model performs forward and backward passes for each batch, computes gradients, updates weights, and logs batch loss. After each epoch, training and test accuracy/loss are recorded using the evaluation function.

### Improvements:
**1. Layer Architecture:**
Increased depth by expanding from 2 blocks (3 layers each) to 4 blocks with varying numbers of layers. This allows the model to learn more complex features. The first block has 4 layers (channels 3->64), second has 3 layers (channels 64->128), third has 2 layers(channels 128->256) and the fourth block has 2 layers (channels 256->512).
**2. Pooling Layers:**
Added max pooling after each convolutional block to reduce spatial dimensions and retain important features, and used average pooling before the final classification layer for better generalization.
**3. Weight Initialization:**
Used Xavier (Glorot) initialization to maintain the variance of activations through layers, helping prevent vanishing or exploding gradients.
**4. Dropout Regularization:**
Applied dropout after convolution layers in the intermediate blocks and as well as the output block to reduce overfitting by randomly deactivating neurons during training.
**5. Optimizer:**
Switched to the Adam optimizer for efficient and adaptive learning rate adjustments during training.
**6. Learning Rate Strategy:**
Implemented the OneCycleLR scheduler with Adam to gradually increase and then decrease the learning rate, promoting faster convergence and better generalization.
**7. Label Smoothing:**
Introduced label smoothing in the loss function to reduce model overconfidence and improve generalization by softening the target labels.
**8. Data Augmentation:** 
- RandomCrop: Randomly crops a portion of the image to encourage the model to focus on different parts of the object.
- RandomHorizontalFlip: Flips the image horizontally with a probability to make the model invariant to object orientation.
- RandomErasing: Randomly removes a rectangular region in the image to simulate occlusion and improve robustness.
- Cutout: Masks out square regions of the image by zeroing them, forcing the model to rely on less prominent features.

### Hyperparameter for the best model:
1.	Batch size: 128
2.	Number of Epochs: 100
3.	Optimizer: Adam
4.	Learning Rate Scheduler: OneCycleLR (max learning rate = 0.005 and annealing strategy = Cosine)
5.	Loss function: Cross Entropy with label smoothing (0.05)
6.	Data Augmentation: RandomCrop(32),RandomHorizontalFlip, RandomErasing(p=0.5), Cutout(n_holes = 1, length=8)
7.	Dropout Rate: Intermediate-0.2, Output- 0.5

### Results:
The improved architecture gave us an accuracy of 91.75%.

### Plots:

 
### Previous Implementations:
I have tried plenty of iterations on each of the hyperparameter to get the optimal value. For example, I have tried various values of the dropout probability for both the intermediate and output layers before selecting the final one. Similarly, I have various batch sizes, pool types, kernel sizes, number of layers, number of blocks, number of output channels, different optimizers, etc. I have also tried advanced layer architectures including skip connections, bottleneck layers, and data augmentation techniques like cutmix, mixup, random augmentation, and auto-augmentation. However, due to the relatively small size of the model, these advanced methods instead led to reduced performance. As a result, I chose to retain only the techniques that consistently improved accuracy in the final implementation.

### Conclusion:
In this project, we designed and trained a deep convolutional neural network for CIFAR-10 image classification, implementing several improvements over a basic model architecture. By incorporating advanced data augmentation techniques (such as RandomCrop, Cutout, and RandomErasing) dropout, Xavier weight initialization, and OneCycleLR with the Adam optimizer, we were able to significantly enhance the model’s generalization ability and overall performance. The use of label smoothing further improved model robustness by preventing overconfidence in predictions. Our training and evaluation pipeline was thoroughly monitored using loss and accuracy metrics, and early stopping ensured optimal model selection. Overall, the implemented strategies led to a more accurate and resilient neural network, demonstrating the effectiveness of modern deep learning techniques in image classification tasks.

**Note:** This project was done as part of the coursework during my master's degree. I have implemented the architecture as given.
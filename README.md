# faceswitch

## Run app

``` python mask_detect.py ```

## Datasets

### Dataset 1
- Source: https://github.com/techyhoney/Facemask_Detection/tree/master/dataset
- Total: 4,000 images
- with_mask: 2,000 images
- without_mask: 2,000 images

### Dataset 2
- Source: https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
- Total: 90,500 images
- with_mask: 500 images
- without_mask: 90,000 images

### Dataset 2.1
- Source: Subset of dataset 2
- Total: 6,000 images
- with_mask: 500 images
- without_mask: 5,500 images 

### Dataset 3
- Source: Combined set of dataset 1 and 2.1
- Total: 10,000 images
- with_mask: 2,500 images
- without_mask: 7,500 images 


## Models

### maskdetection
5 convolution layers, with 3 fully connected layers

Model parameters:
- Input size - (96,96,3)
- Learning rate - 0.0005
- Epochs - 100
- Early Stopping - 30 epochs
- Bach size - 32
- Optimiser - Adam
- Loss function - Binary cross-entropy

### resNet50
ResNet50 architecture

Model parameters:
- Input size - (96,96,3)
- Learning rate - 0.0005
- Epochs - 100
- Early Stopping - 30 epochs
- Bach size - 32
- Optimiser - Adam
- Loss function - Binary cross-entropy
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


## Trained Models

### maskdetection_1
Based on https://github.com/techyhoney/Facemask_Detection

Dataset: 1

Model parameters:
- Input size - (96,96,3)
- Learning rate - 0.0005
- Epochs - 100
- Early Stopping - 30 epochs
- Bach size - 32
- Optimiser - Adam
- Loss function - Binary cross-entropy

### resNet50_1
ResNet50 architecture

Dataset: 1

Model parameters:
- Input size - (96,96,3)
- Learning rate - 0.0005
- Epochs - 100
- Early Stopping - 30 epochs
- Bach size - 32
- Optimiser - Adam
- Loss function - Binary cross-entropy

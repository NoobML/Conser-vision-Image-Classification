# Conser-vision-Image-Classification


# Wildlife Species Classification - Taï National Park

## Overview

This repository contains my solution for the Taï National Park species classification challenge. The goal of this competition was to build a model that accurately classifies animals captured in camera trap images into one of seven species categories or as blank images. My final model achieved a rank of **188** with a **log loss score of 2.0798**.

## Dataset

The dataset consists of images from camera traps in Taï National Park, along with metadata that includes:

- **id**: Unique identifier for each image
- **filepath**: Path to the image file
- **site**: The location where the image was captured (different sites for train and test sets)

### Labels

Each image belongs to one of the following classes:

- **antelope\_duiker**
- **bird**
- **blank** (no animal present)
- **civet\_genet**
- **hog**
- **leopard**
- **monkey\_prosimian**
- **rodent**

## Model Architecture

Unlike many participants who used pretrained models, I designed a **custom CNN architecture** from scratch to minimize classification error. The model consists of:

- **Six convolutional layers** with ReLU activation and L2 regularization.
- **Batch normalization** after each convolutional layer to stabilize training.
- **Max pooling** after early layers to reduce spatial dimensions.
- **Global average pooling** to aggregate feature maps before classification.
- **Fully connected layer** (128 neurons) for final feature processing.
- **Softmax activation** in the output layer to predict class probabilities.

### Model Parameters Table

| Layer                   | Type               | Output Shape         | Param # |
| ----------------------- | ------------------ | -------------------- | ------- |
| image\_input            | InputLayer         | (None, 224, 224, 3)  | 0       |
| conv2d                  | Conv2D             | (None, 222, 222, 32) | 896     |
| batch\_normalization    | BatchNormalization | (None, 222, 222, 32) | 128     |
| activation              | Activation         | (None, 222, 222, 32) | 0       |
| conv2d\_1               | Conv2D             | (None, 222, 222, 64) | 18496   |
| batch\_normalization\_1 | BatchNormalization | (None, 222, 222, 64) | 256     |
| activation\_1           | Activation         | (None, 222, 222, 64) | 0       |
| conv2d\_2               | Conv2D             | (None, 220, 220, 64) | 36928   |
| ...                     | ...                | ...                  | ...     |
| dense                   | Dense              | (None, 128)          | 32896   |
| dense\_1                | Dense              | (None, 8)            | 1032    |

## Training Strategy

- **Data Loading**: A custom data generator was implemented to efficiently load images and labels in batches.
- **Preprocessing**: Images were resized to 224x224 and normalized.
- **Loss Function**: Binary cross-entropy (log loss), as the competition required probability outputs.
- **Optimizer**: Adam optimizer with an initial learning rate of **0.2**.
- **Learning Rate Reduction**: `ReduceLROnPlateau` was used to reduce the learning rate if validation loss plateaued.
- **Checkpointing**: `ModelCheckpoint` was used to save the best-performing model weights.
- **Validation Strategy**: The dataset was split into 80% training and 20% validation, ensuring proper generalization.

## Conclusion

This project demonstrates the effectiveness of a custom CNN model for species classification in camera trap images. While pretrained models could offer better performance, designing a model from scratch allowed for a deeper understanding of CNN architecture and its impact on classification accuracy. Future improvements could include experimenting with data augmentation, ensembling multiple models, and integrating additional metadata features to enhance predictions. Additionally, leveraging transfer learning in combination with a custom architecture could provide further improvements while maintaining the benefits of a handcrafted model.




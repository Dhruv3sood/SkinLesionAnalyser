# Deep Learning for Skin Lesion Detection Using HAM10000 Dataset

## Overview

This deep learning project, crafted for the 'Computing for Medicine' course, employs the HAM10000 dataset to develop a cutting-edge skin lesion detection system. Our goal is to harness the power of machine learning to accurately classify skin lesions, focusing on the critical distinction between benign and malignant types.

## Dataset

The HAM10000 dataset is a comprehensive collection of over 10,000 dermatoscopic images, primarily used for training algorithms in skin lesion detection. It includes a wide range of skin conditions, notably melanoma, nevi, and seborrheic keratoses. Each image is accompanied by detailed clinical metadata such as lesion type, patient age, and sex, making it a valuable resource for developing advanced machine learning models in medical diagnostics.

## Objective

To create an efficient deep learning model for the precise classification of skin lesions, particularly differentiating between benign and malignant lesions.

## Data Preprocessing

- Image Resizing: Conforming images to 224x224 pixels for MobileNet compatibility.
- Normalization: Standardizing pixel values for optimal model training.
- Data Augmentation: Incorporating techniques like random flipping, Gaussian noise, brightness adjustment, and zooming/rotation.

## Model Architecture

The project uses the MobileNet architecture, a streamlined CNN variant known for its efficiency and accuracy in image processing tasks. MobileNet is particularly suitable for handling large datasets like HAM10000, offering a balance between computational resource demands and performance.

## Training

The training of our model is a meticulous process, emphasizing pattern recognition in various skin lesions. We employ data augmentation techniques like random flipping, brightness adjustments, and Gaussian noise, alongside standard preprocessing, to enhance the model's generalization ability and robustness in real-world applications.

The repository includes all relevant code. Contributions towards enhancing model performance are highly encouraged.

# Emotion Recognition Using Computer Vision (CNN)
A real-time facial emotion recognition system that classifies Happy and Sad expressions using a Convolutional Neural Network trained on the FER2013 dataset. The model is deployed with interface for real-time detection from a webcam feed.

## Overview
This project implements an end-to-end emotion recognition pipeline, from raw image preprocessing to a deployed, real-time desktop application. It was developed as part of my undergraduate thesis in Computer Engineering at Bayero University Kano, which I wish to extend to health-related classification problems as my area of research interest.

## Features
1. CNN-based classification trained from scratch on the FER2013 dataset
2. Real-time face detection using OpenCV Haar Cascades
3. Live webcam inference with a responsive desktop GUI (PySide/QML)
4. TorchScript deployment for optimized, portable model inference
5. Modular codebase separating training, export, and inference logic

## Table1: Tools, Libraries and Frameworks
| *Tools* | *Used* |
|-------------|-------------------|
| PyTorch | Model Training |
| OpenCV | Image Processing |
| Pyside/Qt | GUI  |
|QML |UI|
| Numpy and Pandas | Data Handling |

## Dataset
The FER2013 dataset (Facial Expression Recognition 2013) was used as the primary dataset for this project. It is a widely recognized benchmark dataset in the field of facial emotion recognition and is commonly used for training and evaluating deep learning models. The dataset consists of grayscale facial images with a resolution of 48 × 48 pixels. The images represent a variety of facial expressions captured under different conditions, including variations in lighting, facial orientation, and image quality.

## Table 2: FER2013 Dataset Distribution
| **Emotion** | **Training** | **Public Test** | **Private Test** | **Total** |
|---------|----------|-------------|--------------|------|
| **Happy** | 8989 | 895 | 879 | 10763 |
| **Sad** | 6077 | 653 | 594 | 7324 |
| **Total** | 15066 | 1548 | 1473 | 18087 |

**Data Source:** https://www.kaggle.com/datasets/msambare/fer2013

**Data Preprocessing:** 
- Image Resizing
- Normalization
- Label encoding
- Data Augmentation
- Transforming PIL to Tensor

## Model Architecture
A Convolutional Neural Network built from scratch in PyTorch, consisting of stacked convolution + pooling layers, flatten layer followed by fully connected classification layers. 
The architecture was tuned specifically for the reduced-complexity binary classification task (Happy vs. Sad).

## Table 3: Model Performance Metrics Table
| Metric | Score |
|---------|------|
| Accuracy | 91.41% 
| Misclassification | 8.59% |
| Precision | 94% |
| Recall | 90.9% |
| F1-Score | 92.4% |
| Specificit | 92% | 

<img width="480" height="400" alt="image" src="https://github.com/user-attachments/assets/b5533db3-8e08-488f-9db1-d85418afb704" />

*Figure 1: confusion matrix*

<img width="1000" height="500" alt="image" src="https://github.com/user-attachments/assets/83c76b40-4ee8-4b43-ac28-07de78bb27cf" />

*Figure 2: Training Vs Validation Loss and Accuracy graph*

## How To Run

1. Clone the Repository
   git clone `https://github.com/faisalkh523/Emotion_Recognition_Using_Computer_Vision-CNN.git`

2. Install dependencies
   Pip install `write name of the library`

3. Run files
   - Run the application:  `Python Run.py`
   - Retrain model: `python trainexportfull.py`




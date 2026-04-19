# Fake vs Real Image Detection

## Overview
This project is a Deep Learning-based web application that detects whether an image is Real or AI/Edited (Fake) using a trained neural network model.


## Model Used
- MobileNetV2 (Transfer Learning)
- TensorFlow / Keras
- Binary Classification (Real vs Fake)


## Dataset
Custom image dataset split into:
- Training set  
- Validation set  

(Dataset is not included as the trained model is used for prediction)


## Features
- Upload image from device  
- Instant prediction (Real / Fake)  
- Confidence score display  
- Visual charts for analysis  


## Evaluation Metrics
- Accuracy  
- Loss  
- Validation Accuracy  
- Training vs Validation Graph  


## How to Run Locally

### Install dependencies
```bash
pip install -r Requirements.txt
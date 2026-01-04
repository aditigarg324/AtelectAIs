# Atelectasis Detection Using Deep Learning

This project presents a deep learning–based system for the detection of Atelectasis from
chest X-ray images. The system integrates a convolutional neural network with a Flask-based
web interface to allow image upload and real-time prediction.

The project was developed as part of a college-level academic curriculum with emphasis on
medical image analysis, reproducibility, and clean software practices.

## Project Overview
Atelectasis is a common abnormality observed in chest radiographs. Early detection is
important for clinical assessment. This project aims to automate the detection process
using a trained deep learning model and visualization tools.

## Key Features
- CNN-based Atelectasis classification
- Image preprocessing using PIL
- Performance evaluation using scikit-learn
- Visualization of confusion matrix, ROC curve, and training metrics
- Flask-based web interface for predictions
- Modular and reproducible Python codebase

## Tech Stack
- Python
- Flask
- HTML
- TensorFlow / Keras
- NumPy, Pandas
- Matplotlib, Seaborn
- scikit-learn
- PIL (Pillow)
- tqdm

## Dataset
The dataset used for training and testing is excluded from this repository due to:
- Large file size
- Medical data privacy constraints

Public chest X-ray datasets containing Atelectasis cases may be used to reproduce the results.

## How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt

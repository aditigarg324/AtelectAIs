# Atelectasis Detection Using Deep Learning
This project presents a deep learning–based system for the detection of Atelectasis from chest X-ray images. 
The system integrates a convolutional neural network with a Flask-based web interface to allow image upload and 
real-time prediction.The project was developed as part of a college-level academic curriculum with emphasis on 
medical image analysis, reproducibility, and clean software practices. 

The final model achieved 64.52% classification accuracy on the test set.

## Project Overview
Atelectasis is a common abnormality observed in chest radiographs. Early detection is important for clinical assessment. 
This project aims to automate the detection process using a trained deep learning model and visualisation tools.

## My Contribution
Took primary ownership of the data pipeline — organised and validated 5,606 images from the NIH Chest X-ray dataset (Kaggle) 
into 15 class-wise directories, cleaned and filtered down to 728 model-ready training images. Also drove the overall project 
structure across preprocessing, model training, and evaluation stages.

## Key Features

- CNN-based Atelectasis classification
- Image preprocessing using PIL
- Performance evaluation using scikit-learn
- Visualisation of confusion matrix, ROC curve, and training metrics
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
The model was trained on 5,606 images from the NIH Chest X-ray dataset (Kaggle), cleaned and filtered to 728 validated training images 
across 15 class-wise directories. The dataset is excluded from this repository due to large file size and medical data privacy constraints. 
Public chest X-ray datasets containing Atelectasis cases may be used to reproduce the results.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

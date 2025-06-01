# Emotion Recognition with Deep Learning

This repository contains a Colab notebook (DeepLearningProject.ipynb) that demonstrates an end-to-end pipeline for training and evaluating a convolutional neural network (CNN) to classify facial expressions into four categories: Angry, Happy, Neutral, and Sad. The notebook uses a dataset hosted on Google Drive, performs data preprocessing, implements K-fold cross-validation, and visualizes performance via a confusion matrix.

---

## Table of Contents
- Project Overview
- Dataset
- Prerequisites & Dependencies
- Usage: Running the Notebook
- Notebook Structure
  1. Mounting Google Drive & Unzipping Data
  2. Data Preparation
  3. Dataset Loading & Preprocessing
  4. Model Architecture & Training
  5. K-Fold Cross-Validation
  6. Evaluation & Confusion Matrix
- Result Outputs
- Project Structure
- Acknowledgments

---

## Project Overview

Facial emotion recognition is a classic computer vision problem where an image of a human face is classified into different emotion categories. In this project, a CNN is trained on a small four-class dataset (Angry, Happy, Neutral, Sad) to automatically predict the expressed emotion.

The notebook covers:

1. Connecting to Google Drive to access a zipped dataset.
2. Unzipping and arranging image files into their class folders.
3. Preprocessing images (resizing, normalization).
4. Splitting data using K-fold cross-validation for more robust evaluation.
5. Building a simple CNN with Keras (TensorFlow backend).
6. Training the model and tracking metrics.
7. Visualizing performance using a confusion matrix.

---

## Dataset

The dataset used here is expected to live on your Google Drive in the following form:

/MyDrive/  
├─ DLCV_SS25_Dataset.zip        # Main archive (optional; you can replace with individual zips for each class)  
├─ Angry.zip                    # Contains images for the “Angry” class  
├─ Happy.zip                    # Contains images for the “Happy” class  
├─ Neutral.zip                  # Contains images for the “Neutral” class  
└─ Sad.zip                      # Contains images for the “Sad” class  

Each .zip file should contain only JPEG/PNG images of faces for that particular emotion. When unzipped, you’ll end up with four folders at the notebook’s working directory:

content/  
├─ Angry/  
├─ Happy/  
├─ Neutral/  
└─ Sad/  

Note: The notebook file itself refers to /content/drive/MyDrive/ as the base path when unzipping. Adjust these paths if your dataset is stored elsewhere.

---

## Prerequisites & Dependencies

This project is set up to run in Google Colab, but you can also run it locally if you have the required libraries installed. Below are the main dependencies:

- Python 3.7+  
- TensorFlow 2.x (with tensorflow.keras)  
- PyTorch (used minimally for data structures; the model itself uses Keras)  
- scikit-learn (train_test_split, clustering, metrics)  
- OpenCV (cv2) – for reading/resizing images  
- Pillow (PIL.Image) – alternative image I/O  
- NumPy  
- Matplotlib – for plotting and displaying the confusion matrix  
- Google Colab utilities (google.colab.drive)  
- Other utilities: shutil, os, random, hashlib, datetime, collections.deque  

If you run locally, you can install them via:

pip install tensorflow==2.* torch scikit-learn opencv-python pillow matplotlib

---

## Usage: Running the Notebook

1. Upload the notebook (DeepLearningProject.ipynb) to your Google Drive or open it directly in Colab.

2. Ensure the dataset .zip files (DLCV_SS25_Dataset.zip, Angry.zip, Happy.zip, Neutral.zip, Sad.zip) are present under MyDrive/ (or update the paths in the notebook to point to wherever you stored them).

3. Open in Colab  
   - Go to https://colab.research.google.com → Upload → Select DeepLearningProject.ipynb from Drive.

4. Run cells in order. The high-level sequence is:  
   1. Mount Google Drive  
   2. Unzip dataset archives  
   3. Prepare folders and preprocessing code  
   4. Load images into NumPy arrays (with corresponding labels)  
   5. Define and compile the CNN  
   6. Train with K-fold splitting  
   7. Evaluate and plot a confusion matrix  

5. Inspect results: After training, you’ll see training/validation curves and a confusion matrix on the test splits.

---

## Notebook Structure

Below is a brief walkthrough of each major section in DeepLearningProject.ipynb.

### 1. Mounting Google Drive & Unzipping Data

- from google.colab import drive  
  drive.mount('/content/drive')  

  Mounts your Google Drive at /content/drive/.

- !unzip /content/drive/MyDrive/DLCV_SS25_Dataset.zip  
  !unzip /content/drive/MyDrive/Angry.zip  
  !unzip /content/drive/MyDrive/Happy.zip  
  !unzip /content/drive/MyDrive/Neutral.zip  
  !unzip /content/drive/MyDrive/Sad.zip  

  Unzips the four emotion folders into /content/.

### 2. Data Preparation

- Directory Creation  
  Creates subfolders (e.g., dataset/Angry/, dataset/Happy/, etc.) if they don’t exist.  
- Moving/Copying Images  
  Uses shutil to move or copy each class folder into a unified dataset/ directory.

### 3. Dataset Loading & Preprocessing

- Image I/O  
  Uses OpenCV (cv2.imread) and/or PIL (Image.open) to read images.  
- Image Resizing  
  Every image is resized (e.g., 48×48 or 64×64 pixels) and normalized to [0,1].  
- Label Encoding  
  Assigns a numeric label (e.g., 0 = Angry, 1 = Happy, etc.) and one-hot encodes via to_categorical.  
- Train/Test Split  
  Uses sklearn.model_selection.train_test_split (though the primary evaluation uses a K-fold approach).

### 4. Model Architecture & Training

- Framework  
  Built with TensorFlow’s Keras API (tensorflow.keras.models.Sequential).  
- Layers  
  - Convolutional layers (Conv2D, MaxPooling2D)  
  - Flattening  
  - Dense layers with ReLU activations  
  - Final Dense layer with softmax (4 output neurons)  
- Compilation  
  - Optimizer: Adam (or similar)  
  - Loss: categorical_crossentropy  
  - Metrics: accuracy  
- Early Stopping  
  Uses tensorflow.keras.callbacks.EarlyStopping to prevent overfitting.

### 5. K-Fold Cross-Validation

- Setup  
  - sklearn.model_selection.KFold (e.g., n_splits=5)  
  - Iterates through each fold, training on (k−1)/k of data and validating on 1/k.  
- Training Loop  
  For each fold:  
  1. Split X and y into train/validation sets.  
  2. Instantiate a fresh model.  
  3. Fit the model for a fixed number of epochs (with early stopping).  
  4. Save validation accuracy and loss metrics per fold.  
- Aggregated Metrics  
  After all folds finish, the notebook prints average accuracy and loss across folds.

### 6. Evaluation & Confusion Matrix

- Final Test Set  
  After K-fold, the notebook typically keeps aside a small test set (or uses the last fold as “test”).  
- Predictions  
  - Uses model.predict(X_test) to obtain probabilities.  
  - Converts them to class indices with np.argmax(...).  
- Confusion Matrix  
  - Builds a confusion matrix via sklearn.metrics.confusion_matrix.  
  - Visualizes it with sklearn.metrics.ConfusionMatrixDisplay and matplotlib.pyplot.  
- Model Visualization  
  - Optionally calls plot_model(model, to_file="cnn_model.png", show_shapes=True, show_layer_names=True) to save an image of the model architecture.

---

## Result Outputs

1. Training & Validation Curves  
   - Plots of loss and accuracy across epochs for each fold (if cell is enabled).  
2. Average Cross-Validation Score  
   - Printed summary (e.g., “Mean CV Accuracy: 92.5%”).  
3. Confusion Matrix on Test Set  
   - A heatmap showing true vs. predicted classes (Angry, Happy, Neutral, Sad).  
4. Saved Model Diagram  
   - cnn_model.png (Keras visualization of the CNN)

---

## Project Structure

(root)
├─ DeepLearningProject.ipynb # Main Colab notebook
├─ README.md # (This file)
├─ requirements.txt # (Optional; list of pip packages)
├─ cnn_model.png # Generated model architecture plot
└─ (dataset folders after unzip)
├─ Angry/
├─ Happy/
├─ Neutral/
└─ Sad/


- If you choose to unzip locally (instead of in Colab), make sure these folders exist in your working directory.

---

## Acknowledgments

- Dataset Source: Adapted (or provided by) DL/CV course materials (e.g., “DLCV_SS25_Dataset”).  
- Libraries & Tutorials:  
  - TensorFlow Keras Documentation  
  - scikit-learn K-Fold Cross-Validation  
  - ConfusionMatrixDisplay  

Feel free to update this README with any additional details (e.g., hyperparameter values, links to related research papers, etc.). Good luck and happy coding!





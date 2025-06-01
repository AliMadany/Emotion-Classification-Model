# Emotion Recognition with Deep Learning

This repository contains a Colab notebook (`DeepLearningProject.ipynb`) that demonstrates an end-to-end pipeline for training and evaluating a convolutional neural network (CNN) to classify facial expressions into four categories: **Angry**, **Happy**, **Neutral**, and **Sad**. The notebook uses a dataset hosted on Google Drive, performs data preprocessing, implements K-fold cross-validation, and visualizes performance via a confusion matrix.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Prerequisites & Dependencies](#prerequisites--dependencies)  
- [Usage: Running the Notebook](#usage-running-the-notebook)  
- [Notebook Structure](#notebook-structure)  
  1. [Mounting Google Drive & Unzipping Data](#1-mounting-google-drive--unzipping-data)  
  2. [Data Preparation](#2-data-preparation)  
  3. [Dataset Loading & Preprocessing](#3-dataset-loading--preprocessing)  
  4. [Model Architecture & Training](#4-model-architecture--training)  
  5. [K-Fold Cross-Validation](#5-k-fold-cross-validation)  
  6. [Evaluation & Confusion Matrix](#6-evaluation--confusion-matrix)  
- [Result Outputs](#result-outputs)  
- [Project Structure](#project-structure)  
- [Acknowledgments](#acknowledgments)  

---

## Project Overview

Facial emotion recognition is a classic computer vision problem where an image of a human face is classified into different emotion categories. In this project, a CNN is trained on a small four-class dataset (Angry, Happy, Neutral, Sad) to automatically predict the expressed emotion.

The notebook covers:

1. **Connecting to Google Drive** to access a zipped dataset.  
2. **Unzipping** and arranging image files into their class folders.  
3. **Preprocessing** images (resizing, normalization).  
4. **Splitting** data using K-fold cross-validation for more robust evaluation.  
5. **Building** a simple CNN with Keras (TensorFlow backend).  
6. **Training** the model and tracking metrics.  
7. **Visualizing** performance using a confusion matrix.  

---

## Dataset

The dataset used here is expected to live on your Google Drive in the following form:


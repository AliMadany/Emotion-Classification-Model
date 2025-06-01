# Deep Learning Emotion Classification

## Project Overview
This repository contains a deep learning project for classifying facial emotions into four categories: **Angry**, **Happy**, **Sad**, and **Neutral**. The project is implemented as a Jupyter notebook (`DeepLearningProject.ipynb`) and covers the following stages:
1. **Data Preparation**  
   - Connecting to a Google Drive–hosted dataset  
   - Unzipping and organizing image files into class-specific folders  
   - (Optional) Data augmentation and preprocessing  
2. **Modeling (From Scratch)**  
   - Building a convolutional neural network (CNN) in TensorFlow/Keras  
   - Training with k-fold cross-validation (K=4)  
   - Evaluating on validation and test splits  
3. **Modeling (Using Libraries)**  
   - Leveraging higher-level utilities for data loading and augmentation  
   - Repeating training and evaluation experiments using library workflows  
4. **Milestone & Visualization**  
   - Plotting accuracy and loss curves over epochs  
   - Displaying a confusion matrix for final test-set performance  

By the end of this notebook, you will have a trained CNN capable of distinguishing between the four emotion classes and analysis plots showcasing its performance.  

---

## Table of Contents
1. [Dataset](#dataset)  
2. [Prerequisites](#prerequisites)  
3. [Installation](#installation)  
4. [Project Structure](#project-structure)  
5. [Usage](#usage)  
6. [Model Architecture](#model-architecture)  
7. [Training & Evaluation](#training--evaluation)  
8. [Results & Visualizations](#results--visualizations)  
9. [Dependencies](#dependencies)  
10. [Contributing](#contributing)  
11. [License](#license)  

---

## Dataset
- **Source**: The dataset should be stored on Google Drive (or a similar cloud storage) and consists of facial images categorized into four emotion subfolders:  
  - `Angry/`  
  - `Happy/`  
  - `Sad/`  
  - `Neutral/`  
- **Organization**:  
  1. Upload the zipped dataset (containing the four folders) to your Google Drive.  
  2. Mount Google Drive within the notebook to access the data.  
  3. Unzip the archive and merge all images into a single directory structure under `emotions_combined/` (one subfolder per class).  

> **Note:** Images are resized to 224×224 pixels before being fed to the CNN.  

---

## Prerequisites
- A Python 3.7+ environment (tested on Python 3.8)  
- Access to a GPU (recommended, but CPU-only also works; training will be slower)  
- A Google Drive account (if you wish to follow the “Connect to Drive” cells as-is)  

---

## Installation
1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-username/deep-learning-emotion-classification.git
   cd deep-learning-emotion-classification
   ```

2. **Create & activate a virtual environment (optional but recommended)**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate       # Linux/Mac
   venv\Scripts\activate.bat      # Windows
   ```

3. **Install required Python packages**  
   ```bash
   pip install -r requirements.txt
   ```
   _If you don’t have a `requirements.txt`, install the main dependencies manually:_  
   ```bash
   pip install numpy pandas matplotlib tensorflow torch opencv-python pillow scikit-learn
   ```

4. **(Optional) If you plan to use TensorBoard for monitoring**  
   ```bash
   pip install tensorboard
   ```

---

## Project Structure
```
├── DeepLearningProject.ipynb   # Main notebook covering data prep, modeling, evaluation, and visualization
├── requirements.txt            # List of Python dependencies
├── README.md                   # This file
└── data/                       # (Optional) Local folder for unzipped images if not using Drive
    ├── emotions_combined/
    │   ├── Angry/
    │   ├── Happy/
    │   ├── Sad/
    │   └── Neutral/
    └── ...
```
- **DeepLearningProject.ipynb**  
  - Contains all code cells for:  
    1. Mounting Google Drive  
    2. Unzipping & organizing data  
    3. Data pre-processing (resizing, normalization, augmentation)  
    4. Model definition (from scratch)  
    5. Model definition (using higher-level libraries)  
    6. Training loops (with 4-fold cross-validation)  
    7. Evaluation on a held-out test set  
    8. Plotting accuracy/loss and confusion matrix  

- **requirements.txt**  
  - Pin exact library versions used (e.g., `tensorflow==2.10.0`, `torch==1.12.0`, etc.).  

---

## Usage

1. **Open the Notebook**  
   Launch Jupyter (or JupyterLab) from the repository root:  
   ```bash
   jupyter notebook DeepLearningProject.ipynb
   ```
   or
   ```bash
   jupyter lab DeepLearningProject.ipynb
   ```

2. **Connect to Google Drive (if using Drive)**  
   In the first code cell, run:  
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Then point `DATA_PATH` to the mounted folder containing your zipped dataset.

3. **Unzip & Organize Data**  
   Follow the “Unzipping the dataset folder” cells. This will create:
   ```
   /content/emotions_combined/
      ├── Angry/
      ├── Happy/
      ├── Sad/
      └── Neutral/
   ```

4. **Run Data Preparation**  
   - Resize all images to 224×224 pixels.  
   - Optionally apply data augmentation (e.g., flips, rotations).

5. **Run Model Training**  
   - **Part 1 (From Scratch)**  
     - Defines a custom CNN with Conv2D → Pooling → BatchNorm → Dropout → Dense layers.  
     - Compiles with `Adam` optimizer and `sparse_categorical_crossentropy` loss.  
     - Trains for `EPOCHS = 20`, `BATCH_SIZE = 8`, using 4-fold cross-validation.  
   - **Part 2 (Using Libraries)**  
     - Uses `ImageDataGenerator` (or similar) for on-the-fly augmentation and batching.  
     - Repeats training/validation splits for comparison.

6. **Evaluate & Visualize**  
   - Once training is complete, evaluate on a held-out test set.  
   - Plot training vs. validation accuracy/loss curves over epochs.  
   - Compute and display a confusion matrix for final test-set predictions.

---

## Model Architecture

Below is a high-level summary of the CNN architecture defined in **Part 1**:

```
Input: 224×224×3 image
└─ Conv2D (32 filters, 5×5, ReLU)  
   └─ MaxPooling2D (2×2)  
   └─ BatchNormalization  
   └─ Conv2D (16 filters, 7×7, ReLU)  
       └─ MaxPooling2D (2×2)  
       └─ Dropout (0.30)  
       └─ Flatten  
       └─ Dense (64 units, sigmoid)  
       └─ Dense (4 units, softmax)
```

- **Optimizer**: Adam  
- **Loss**: Sparse Categorical Cross-Entropy  
- **Metrics**: Accuracy  

In **Part 2**, you may see a variation that utilizes `tf.keras.preprocessing.image.ImageDataGenerator` pipelines and potentially a different CNN backbone.

---

## Training & Evaluation

- **K-Fold Cross-Validation (K=4)**  
  1. Split combined dataset into 4 folds.  
  2. For each fold:  
     - Use 3 folds for training, 1 fold for validation.  
     - Train for 20 epochs (adjustable).  
     - Record validation accuracy/loss.  
  3. Average validation metrics across folds to estimate generalization.  

- **Final Test Evaluation**  
  - After cross-validation, set aside a separate “test” split (e.g., 10–15% of overall data).  
  - Evaluate the best-performing fold’s model (or the model retrained on all train+val folds) on this test set.  
  - Display a confusion matrix using `sklearn.metrics.ConfusionMatrixDisplay`.  

- **Visualization**  
  - Plot curves showing training vs. validation accuracy and loss for each epoch.  
  - After test evaluation, plot a 4×4 confusion matrix heatmap labeled with the four emotion classes.  

---

## Results & Visualizations

1. **Accuracy & Loss Curves**  
   After running the training cells, you should see two line plots per training session (one for accuracy, one for loss). Typical outputs include:  
   - Training accuracy approaching ~80–90% by epoch 20 (depending on dataset size and augmentation).  
   - Validation accuracy slightly lower, indicating generalization performance.  

2. **Confusion Matrix**  
   A heatmap that shows how often each true emotion class is predicted as each of the four labels. An ideal confusion matrix will have high values along the diagonal (correct predictions) and low off-diagonal values (misclassifications).

   Example (mock layout):

   | Predicted \ True | Angry | Happy | Sad | Neutral |
   |-------------------|-------|-------|-----|---------|
   | **Angry**         | 45    |   3   |  2  |   0     |
   | **Happy**         |  2    | 48    |  1  |   0     |
   | **Sad**           |  1    |   2   | 47  |   1     |
   | **Neutral**       |  0    |   1   |  2  |  46     |

---

## Dependencies

Below is a non-exhaustive list of Python packages used in this project. Exact versions can be found in `requirements.txt`.

- **Core Libraries**  
  - `numpy`  
  - `pandas` (optional; for any tabular logging)  
  - `matplotlib`  
  - `scikit-learn`  

- **Deep Learning Frameworks**  
  - `tensorflow>=2.8.0` (for Keras API)  
  - `torch` (imported but not strictly required for the shown notebook; can be omitted if not used)  

- **Image Processing**  
  - `opencv-python` (cv2)  
  - `Pillow`  

- **Utilities**  
  - `tqdm` (for progress bars; optional)  
  - `google-colab` (if running in Colab; for `drive.mount`)  

---

## Contributing
1. Fork this repository.  
2. Create a new branch (`git checkout -b feature/YourFeature`).  
3. Make your changes and commit them (`git commit -m "Add new data augmentation pipeline"`).  
4. Push to your branch (`git push origin feature/YourFeature`).  
5. Open a Pull Request.  

We welcome improvements, bug fixes, and new features (e.g., additional emotion classes, alternate model architectures, integration with TensorBoard, etc.).

---

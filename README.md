# Task 03: Supervised Classification with a Linear SVM

## Overview

This project demonstrates a **Support Vector Machine (SVM)** using a small, generated binary-classification dataset.

**Important limitation:** despite the original repository title, this code does not classify cat and dog images. It generates numeric feature vectors directly; it does not load image files, extract image features, or train on a labelled image dataset. The class names in this demo are simply **Class A** and **Class B**.

## What the script does

1. Generates two synthetic classes with different numeric feature patterns.
2. Splits the data into training and test sets while preserving the class balance.
3. Uses a pipeline to standardize features and train a linear SVM.
4. Reports held-out test accuracy and a precision/recall/F1 classification report.
5. Predicts the class of one new synthetic example.
6. Displays a confusion matrix for test-set predictions.

## Requirements

- Python 3.10 or newer
- NumPy
- scikit-learn
- Matplotlib

Install dependencies:

```bash
python -m pip install numpy scikit-learn matplotlib
```

## Run

```bash
python SCT_03.py
```

The script prints evaluation metrics and opens a confusion-matrix plot. Close the plot window to finish.

## Concepts demonstrated

- **Supervised learning:** training from examples with known labels.
- **SVM:** learns a decision boundary that separates classes, aiming to maximize the margin between them.
- **Linear kernel:** uses a linear decision boundary in the feature space.
- **Train/test split:** evaluates the model on held-out examples rather than only the examples used for training.
- **Feature scaling:** places numeric features on comparable scales.
- **Confusion matrix:** counts correct and incorrect predictions for each class.
- **Precision, recall and F1-score:** complementary measures of classification performance.

## Limitations and next steps

- The data is synthetic, so the accuracy is not evidence of performance on real images or real-world data.
- The model does not receive pixels, image embeddings, or image descriptors.
- To build a genuine cat-versus-dog image classifier, use a properly labelled image dataset, define image preprocessing or feature extraction, keep train/test images separate, and evaluate the model on held-out images. A CNN or a suitable pretrained image model is a more direct approach for that task.

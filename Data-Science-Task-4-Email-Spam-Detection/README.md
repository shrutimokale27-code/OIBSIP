# Email Spam Detection with Machine Learning

## Project Overview

This project detects whether an SMS or email message is Spam or Ham (Not Spam) using Machine Learning.

The project uses Natural Language Processing (NLP) techniques to convert text messages into numerical features and a Naive Bayes classification algorithm to classify messages.

## Objectives

- Clean and preprocess the text dataset.
- Convert text messages into numerical features.
- Build a Machine Learning classification model.
- Detect Spam and Ham messages.
- Evaluate the model using Accuracy, Precision, Recall and F1-score.
- Visualize the model performance using a Confusion Matrix.
- Test the model with custom messages.

## Dataset

- Dataset: SMS Spam Collection
- Total Messages: 5,572
- Columns: 2
- Target Classes: Ham and Spam
- Duplicate Rows: 403

The dataset contains text messages labelled as either `ham` or `spam`.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- NLTK

## Data Preprocessing

The following steps were performed:

- Loaded the SMS Spam Collection dataset.
- Assigned column names as `label` and `message`.
- Checked missing values.
- Removed duplicate rows.
- Converted labels into numerical values:
  - Ham = 0
  - Spam = 1
- Split the dataset into training and testing sets.

## Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert text messages into numerical features that can be used by the Machine Learning model.

## Machine Learning Model

The project uses:

**Multinomial Naive Bayes**

Naive Bayes is a classification algorithm that works well with text-based datasets and is commonly used for spam detection.

## Model Performance

- Test Dataset: 1,034 messages
- Accuracy: **97.2%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Ham | 0.97 | 1.00 | 0.98 |
| Spam | 1.00 | 0.78 | 0.88 |
| Overall Accuracy | - | - | **0.97** |

## Confusion Matrix

The confusion matrix shows the number of correctly and incorrectly classified Ham and Spam messages.

The model produced:

- Ham correctly classified as Ham: 903
- Ham incorrectly classified as Spam: 0
- Spam incorrectly classified as Ham: 29
- Spam correctly classified as Spam: 102

## Custom Message Testing

Two custom messages were tested:

1. "Congratulations! You have won a free prize. Click now!"
   - Prediction: **Spam**

2. "Hi, are we meeting for the project today?"
   - Prediction: **Ham**

## Project Structure

```text
Data-Science-Task-4-Email-Spam-Detection/
│
├── SMSSpamCollection
├── email_spam_detection.py
└── README.md
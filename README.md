# Spam Email Detection

## Overview

This project aims to build a machine learning model to classify email messages as either spam or non-spam (ham). The model uses Natural Language Processing (NLP) techniques to preprocess the text data and a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) for classification.

## Dataset

The dataset used in this project is `spam emails.csv`, which contains two columns:
- `Category`: Label indicating whether the email is spam or ham.
- `Message`: The content of the email message.

## Libraries and Dependencies

This project requires the following Python libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `nltk`
- `wordcloud`
- `tensorflow`
- `scikit-learn`

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn nltk wordcloud tensorflow scikit-learn
```
## Code Description

1. Import Libraries: Import necessary libraries for data exploration, preprocessing, and model building.
2. Load and Preprocess Data:
- Load the dataset and examine its structure.
- Convert email categories to binary labels (1 for spam, 0 for non-spam).
- Balance the dataset by downsampling the majority class (ham).
3. Text Preprocessing:
- Remove stopwords from the email messages.
- Generate word clouds for spam and non-spam emails to visualize the most frequent words.
- Data Preparation:
4. Split the data into training and testing sets.
- Tokenize and pad the text sequences to ensure uniform input length for the model.
5. Model Building:
- Build an LSTM-based Sequential model using TensorFlow/Keras.
- Compile the model with binary cross-entropy loss and Adam optimizer.
- Train the model with early stopping and learning rate reduction callbacks.
6. Evaluation:
- Evaluate the model's performance on the test set.
- Plot training and validation accuracy over epochs.

## Results
The model's performance will be displayed in terms of test loss and accuracy. Training and validation accuracy plots will also be generated to help visualize the model's learning process.

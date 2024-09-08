# Text Classification - NLP: Emotion Detection in Text

## Overview
This project focuses on **text emotion classification** using Natural Language Processing (NLP). The goal is to classify emotions (like anger, joy, sadness, etc.) from text by training a machine learning model on a labeled dataset.

### Dataset
We use the **[Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)**, which contains text samples with corresponding emotion labels.

### Approach
- **Preprocessing**: Tokenization, stopword removal, lemmatization, and vectorization (TF-IDF).
- **Models**: Both traditional ML models (Logistic Regression, SVM) and deep learning models (LSTM, BiLSTM) are implemented using **TensorFlow**, **Keras**, and **Scikit-Learn**.
- **Evaluation**: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

## Libraries
- **Pandas, Numpy**: Data manipulation and computation.
- **TensorFlow, Keras**: Deep learning models.
- **Scikit-learn**: Machine learning algorithms.

## Usage
- **Train**: `python train.py`
- **Predict**: `python predict.py --text "Sample text"`

---

**Author**: Sarah Sufi

 
**GitHub**: sarahsufi

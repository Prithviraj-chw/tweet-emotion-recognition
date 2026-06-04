# Tweet Emotion Recognition

A machine learning web app that detects emotions from text using NLP techniques and a Logistic Regression classifier trained on tweet data. Built with scikit-learn, NLTK, and Streamlit, with model artifacts packed using joblib.

## Features

- Detects 6 emotions: **joy, sadness, anger, fear, love, surprise**
- NLP preprocessing pipeline: lowercasing, punctuation removal, stopword removal
- Bag-of-Words vectorization with `CountVectorizer`
- Logistic Regression classifier (best accuracy among models tested)
- Simple and clean Streamlit web interface

## Project Structure

```
tweet-emotion-recognition/
│
├── sentiment_analysis.ipynb   # Model training & experimentation notebook
├── app.py                     # Streamlit web app
├── train.txt                  # Training dataset
├── model.joblib               # Saved Logistic Regression model
├── vectorizer.joblib          # Saved CountVectorizer
└── emotion_map.joblib         # Saved emotion label mapping
```

## Models Compared

| Model | Vectorizer |
|---|---|
| Naive Bayes | BoW |
| Naive Bayes | TF-IDF |
| Logistic Regression | TF-IDF |
| SVM | TF-IDF |
| **Logistic Regression** ✅ | **BoW** ✅ |
| SVM | BoW |

> Logistic Regression with Bag-of-Words achieved the best accuracy and was selected for deployment.

## Installation

```bash
pip install scikit-learn nltk streamlit joblib pandas numpy
```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, type any text and click **Predict**.

## Tech Stack

- **Python**
- **scikit-learn** — model training & vectorization
- **NLTK** — stopword removal
- **Streamlit** — web interface
- **joblib** — model serialization

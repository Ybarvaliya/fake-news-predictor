# Fake News Predictor - Machine Learning Model

## Overview
This project uses a machine learning approach to predict whether a news article is fake or real. The model is trained on a dataset of labeled news articles and uses natural language processing (NLP) techniques to analyze the text.

## Dataset
The dataset used for this project is a collection of news articles with corresponding labels indicating whether the article is fake or real. 
- Link - (https://www.kaggle.com/c/fake-news/data?select=train.csv)

## Model - Logistic Regression
The model used for this project is a machine learning classifier that takes the text of a news article as input and outputs a prediction of whether the article is fake or real.

## Features
The model uses the following features to make predictions:
- Text of the news article
- Stopwords (common words like "the", "and", etc. that do not add much value to the meaning of the text)

## The model uses the following NLP techniques to preprocess the text:
- Tokenization (splitting the text into individual words)
- Stemming (reducing words to their base form)
- Lemmatization (reducing words to their base form using a dictionary)

## Model Evaluation
The model is evaluated using the following metrics:
- Accuracy
- Precision
- Recall
- F1-score

## Results
- The model achieves an accuracy of 97.9% on the test dataset.

# Code
The code for this project is written in Python and uses the following libraries:
- Pandas for data manipulation and analysis
- NumPy for numerical computations
- Scikit-learn for machine learning
- NLTK for NLP tasks

## Usage
- To use this project, simply run the Fake_News_Predictor.ipynb notebook and follow the instructions.



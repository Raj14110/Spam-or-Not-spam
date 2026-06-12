# Spam or Not Spam Classification

## Overview

This project implements a **Spam vs Not Spam (Ham) classification system** using **Natural Language Processing (NLP)** and **Machine Learning** techniques.

The system classifies SMS messages into:

* **Spam**
* **Not Spam (Ham)**

The project uses two different approaches:
1. **TF-IDF Vectorization**
2. **Word2Vec Embeddings**
and compares multiple machine learning models to determine the best-performing approach.

---

## Goal :
The goal of this project is to build a machine learning system capable of accurately detecting spam messages from normal text.

---

## Dataset

The project uses the **SMS Spam Collection Dataset** containing approximately **5,500+ SMS messages**.

### Labels

* `ham → 0`
* `spam → 1`

### Example

| Message                          | Label |
| -------------------------------- | ----- |
| Ok lar... Joking wif u oni...    | Ham   |
| Congratulations! You won ₹50,000 | Spam  |

---

## Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Pandas
* NumPy
* Scikit-learn
* Gensim
* XGBoost
* LightGBM
* Matplotlib

---

## Project Workflow

### 1. Data Preprocessing

* Loaded SMS dataset
* Encoded labels:

  * `ham → 0`
  * `spam → 1`
* Applied train-test split using **stratification** to preserve class balance.

---

### 2. Feature Engineering

#### TF-IDF Vectorization

Used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text messages into numerical vectors.

Configuration:

```python
max_features = 5000
stop_words = 'english'
```

#### Word2Vec Embeddings

Trained a **Word2Vec model** using Gensim to generate dense vector embeddings for words.

Configuration:

```python
vector_size = 100
window = 5
min_count = 1
```

Sentence embeddings were created by averaging word embeddings.

---

## Models Used

### For TF-IDF 

* Multinomial Naive Bayes
* Logistic Regression

### For Word2Vec 

* Random Forest Classifier
* XGBoost Classifier
* LightGBM Classifier

---

## Model Performance

| Model                   | Accuracy   |
| ----------------------- | ---------- |
| Multinomial Naive Bayes | **97.43%** |
| Logistic Regression     | 96.53%     |
| XGBoost                 | 95.39%     |
| LightGBM                | 95.27%     |
| Random Forest           | 94.85%     |

### Best Performing Model

**Multinomial Naive Bayes** achieved the highest accuracy using TF-IDF features with **97.43% accuracy**.

---

## Evaluation Metrics

The models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1 Score
* Classification Report

---

## Project Structure

```text
Spam-or-Not-spam/
│── spam_classification.ipynb
│── SMSSpamCollection
│── README.md
│── requirements.txt
```




## Author

**Aritra Mandal**

GitHub: https://github.com/Raj14110

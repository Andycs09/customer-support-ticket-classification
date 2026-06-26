# Customer Support Ticket Classification using BGE Embeddings, HDBSCAN & LinearSVC

A machine learning pipeline for automatically classifying enterprise customer support tickets into issue categories and predicting ticket priority using semantic embeddings.
Run in sequence
cluster.py
ticket_final.py
predict.py
---

#  How to Run

Run the scripts in the following order.

## 1. Cluster the Dataset

```bash
python cluster.py
```

This script:

* Cleans and preprocesses ticket descriptions
* Generates BGE embeddings
* Reduces dimensions using UMAP
* Performs HDBSCAN clustering
* Validates clusters
* Generates the final training dataset

Outputs:

```
ticket_training_dataset.csv
ticket_embeddings.npy
training_indices.npy
```

---

## 2. Train the Models

```bash
python training_final.py
```

This script trains two independent LinearSVC models.

### Category Classifier

Predicts categories such as:

* Software Issue
* Hardware Issue
* Network Issue
* Security Issue
* Account Access Issue
* Data Recovery
* etc.

### Priority Classifier

Predicts

* Low
* Medium
* High

Outputs:

```
linear_svc_ticket_classifier.pkl
priority_classifier.pkl
ticket_labels.pkl
priority_labels.pkl
```

---

## 3. Run Prediction

```bash
python predict.py
```

Example

```
Company:
Sony

Ticket:
Laptop does not power on after sudden shutdown.

Human Priority:
High
```

Output

```
Category            : Hardware Issue

Predicted Priority  : Low

Decision            : Priority Review Required
```

---

# Project Pipeline

```
Ticket Dataset
      │
      ▼
Data Cleaning
      │
      ▼
BGE Embeddings
      │
      ▼
UMAP
      │
      ▼
HDBSCAN
      │
      ▼
Cluster Validation
      │
      ▼
Manual Cluster Labelling
      │
      ▼
Category Classifier
      │
      ▼
Priority Classifier
      │
      ▼
Prediction Engine
```

---

# Cluster Validation Results

```
Total Tickets          : 8469

Clusters Found         : 29

Noise Tickets          : 125 (1.48%)

Silhouette Score       : 0.735

Final Training Samples : 7116
```

---

# Embedding Comparison

BGE embeddings were compared against a TF-IDF baseline.

| Method             | Accuracy   | Precision  | Recall     | Weighted F1 |
| ------------------ | ---------- | ---------- | ---------- | ----------- |
| **BGE Embeddings** | **98.64%** | **98.66%** | **98.64%** | **98.63%**  |
| **TF-IDF**         | 97.61%     | 96.82%     | 97.61%     | 97.16%      |

BGE embeddings achieved the best performance and were selected for the final implementation.

---

# Final Model Performance

## Category Classification

| Metric      | Score      |
| ----------- | ---------- |
| Accuracy    | **98.64%** |
| Precision   | **98.66%** |
| Recall      | **98.64%** |
| Weighted F1 | **98.63%** |

---

## Priority Classification

| Metric      | Score      |
| ----------- | ---------- |
| Accuracy    | **85.43%** |
| Precision   | **85.45%** |
| Recall      | **85.43%** |
| Weighted F1 | **85.44%** |

---

# Technologies Used

* Python
* Sentence Transformers
* BAAI/bge-base-en-v1.5
* UMAP
* HDBSCAN
* Scikit-Learn
* LinearSVC
* NumPy
* Pandas
* Joblib

---

# Repository Structure

```
Customer-Support-Ticket-Classification/

│
├── cluster.py
├── training_final.py
├── compare_embeddings.py
├── predict.py
│
├── ticket_training_dataset.csv
├── ticket_embeddings.npy
├── training_indices.npy
│
├── linear_svc_ticket_classifier.pkl
├── priority_classifier.pkl
├── ticket_labels.pkl
├── priority_labels.pkl
│
├── requirements.txt
└── README.md
```

---

# Installation


Install dependencies

```bash
pip install -r requirements.txt
```

---

# Future Work

* Explainable AI (XAI)
* Confidence calibration
* Company-specific routing
* Active learning
* Multilingual support

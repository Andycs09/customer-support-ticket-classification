"""
train_classifier.py
===================

PART 1

CATEGORY CLASSIFIER

This script trains the final Category Classification model
using BGE embeddings generated during clustering.

Input Files
-----------
ticket_training_dataset.csv
ticket_embeddings.npy
training_indices.npy

Output Files
------------
linear_svc_ticket_classifier.pkl
ticket_labels.pkl

Evaluation
----------
Accuracy
Precision
Recall
Weighted F1
Classification Report
Confusion Matrix
"""

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 70)
print("LOADING DATA")
print("=" * 70)

training_df = pd.read_csv(
    "ticket_training_dataset.csv"
)

embeddings = np.load(
    "ticket_embeddings.npy"
)

training_indices = np.load(
    "training_indices.npy"
)

# =====================================================
# BUILD TRAINING MATRIX
# =====================================================

print("\nBuilding BGE feature matrix...")

X = embeddings[training_indices]

y = training_df["cluster_label"].values

print(f"Feature Shape : {X.shape}")
print(f"Labels        : {len(y)}")
print(f"Classes       : {len(np.unique(y))}")

assert len(X) == len(y)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

print("\nCreating train/test split...")

(
    X_train,
    X_test,
    y_train,
    y_test
) = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# =====================================================
# TRAIN CLASSIFIER
# =====================================================

print("\n" + "=" * 70)
print("TRAINING CATEGORY CLASSIFIER")
print("=" * 70)

category_classifier = LinearSVC(
    C=1.0,
    random_state=42
)

category_classifier.fit(
    X_train,
    y_train
)

print("Training Complete.")

# =====================================================
# PREDICTIONS
# =====================================================

print("\nRunning predictions...")

predictions = category_classifier.predict(
    X_test
)

# =====================================================
# EVALUATION
# =====================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
    zero_division=0
)

# =====================================================
# RESULTS
# =====================================================

print("\n")
print("=" * 70)
print("CATEGORY CLASSIFIER RESULTS")
print("=" * 70)

print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"Weighted F1   : {weighted_f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# =====================================================
# SAVE CATEGORY MODEL
# =====================================================

print("\nSaving category classifier...")

joblib.dump(
    category_classifier,
    "linear_svc_ticket_classifier.pkl"
)

print("Saved:")
print("linear_svc_ticket_classifier.pkl")

# =====================================================
# SAVE CATEGORY LABELS
# =====================================================

category_labels = sorted(
    training_df["cluster_label"].unique()
)

joblib.dump(
    category_labels,
    "ticket_labels.pkl"
)

print("ticket_labels.pkl")

# =====================================================
# CLASS DISTRIBUTION
# =====================================================

print("\nCategory Distribution\n")

print(
    training_df["cluster_label"]
    .value_counts()
    .sort_values()
)

# =====================================================
# SUMMARY
# =====================================================

print("\n")
print("=" * 70)
print("CATEGORY MODEL SUMMARY")
print("=" * 70)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")
print(f"Classes          : {len(category_labels)}")

print(f"\nAccuracy         : {accuracy:.4f}")
print(f"Precision        : {precision:.4f}")
print(f"Recall           : {recall:.4f}")
print(f"Weighted F1      : {weighted_f1:.4f}")

print("\nCategory classifier training complete.")

print("\nContinue with Part 2 to train the Priority Classification model.")

# =====================================================
# PART 2
# PRIORITY CLASSIFIER
# =====================================================

print("\n")
print("=" * 70)
print("PRIORITY CLASSIFIER")
print("=" * 70)

# =====================================================
# BUILD TRAINING MATRIX
# =====================================================

X_priority = embeddings[training_indices]

y_priority = training_df["Priority"].values

print(f"Feature Shape : {X_priority.shape}")
print(f"Labels        : {len(y_priority)}")
print(f"Classes       : {len(np.unique(y_priority))}")

assert len(X_priority) == len(y_priority)

# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

(
    X_train_p,
    X_test_p,
    y_train_p,
    y_test_p
) = train_test_split(
    X_priority,
    y_priority,
    test_size=0.30,
    random_state=42,
    stratify=y_priority
)

print(f"\nTraining Samples : {len(X_train_p)}")
print(f"Testing Samples  : {len(X_test_p)}")

# =====================================================
# TRAIN PRIORITY MODEL
# =====================================================

print("\nTraining Priority Classifier...")

priority_classifier = LinearSVC(
    C=1.0,
    random_state=42
)

priority_classifier.fit(
    X_train_p,
    y_train_p
)

print("Training Complete.")

# =====================================================
# PREDICTIONS
# =====================================================

priority_predictions = priority_classifier.predict(
    X_test_p
)

# =====================================================
# METRICS
# =====================================================

priority_accuracy = accuracy_score(
    y_test_p,
    priority_predictions
)

priority_precision = precision_score(
    y_test_p,
    priority_predictions,
    average="weighted",
    zero_division=0
)

priority_recall = recall_score(
    y_test_p,
    priority_predictions,
    average="weighted",
    zero_division=0
)

priority_f1 = f1_score(
    y_test_p,
    priority_predictions,
    average="weighted",
    zero_division=0
)

# =====================================================
# RESULTS
# =====================================================

print("\n")
print("=" * 70)
print("PRIORITY CLASSIFIER RESULTS")
print("=" * 70)

print(f"Accuracy      : {priority_accuracy:.4f}")
print(f"Precision     : {priority_precision:.4f}")
print(f"Recall        : {priority_recall:.4f}")
print(f"Weighted F1   : {priority_f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test_p,
        priority_predictions,
        zero_division=0
    )
)

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test_p,
        priority_predictions
    )
)

# =====================================================
# SAVE PRIORITY MODEL
# =====================================================

print("\nSaving priority classifier...")

joblib.dump(
    priority_classifier,
    "priority_classifier.pkl"
)

print("Saved:")
print("priority_classifier.pkl")

# =====================================================
# SAVE PRIORITY LABELS
# =====================================================

priority_labels = sorted(
    training_df["Priority"].unique()
)

joblib.dump(
    priority_labels,
    "priority_labels.pkl"
)

print("priority_labels.pkl")

# =====================================================
# PRIORITY DISTRIBUTION
# =====================================================

print("\nPriority Distribution\n")

print(
    training_df["Priority"]
    .value_counts()
    .sort_values()
)

# =====================================================
# FINAL SUMMARY
# =====================================================

print("\n")
print("=" * 70)
print("FINAL TRAINING SUMMARY")
print("=" * 70)

print("\nCATEGORY CLASSIFIER")

print(f"Accuracy      : {accuracy:.4f}")
print(f"Precision     : {precision:.4f}")
print(f"Recall        : {recall:.4f}")
print(f"Weighted F1   : {weighted_f1:.4f}")

print("\nPRIORITY CLASSIFIER")

print(f"Accuracy      : {priority_accuracy:.4f}")
print(f"Precision     : {priority_precision:.4f}")
print(f"Recall        : {priority_recall:.4f}")
print(f"Weighted F1   : {priority_f1:.4f}")

print("\nSaved Models")

print("✓ linear_svc_ticket_classifier.pkl")
print("✓ priority_classifier.pkl")

print("\nSaved Labels")

print("✓ ticket_labels.pkl")
print("✓ priority_labels.pkl")

print("\nTraining Complete.")
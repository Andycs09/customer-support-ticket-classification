"""
compare_embeddings.py
=====================

PART 1

BGE Embeddings + LinearSVC

This script compares BGE embeddings against a TF-IDF baseline.
Part 1 trains and evaluates the BGE classifier.

Part 2 will:
    - Train TF-IDF + LinearSVC
    - Compare both methods
"""

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
    confusion_matrix
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
# BUILD BGE TRAINING MATRIX
# =====================================================

print("\nBuilding BGE feature matrix...")

X_bge = embeddings[training_indices]

y = training_df["cluster_label"].values

print(f"Feature Shape : {X_bge.shape}")
print(f"Labels        : {len(y)}")
print(f"Classes       : {len(np.unique(y))}")

assert len(X_bge) == len(y)

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
    X_bge,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# =====================================================
# TRAIN BGE CLASSIFIER
# =====================================================

print("\n" + "=" * 70)
print("TRAINING BGE + LinearSVC")
print("=" * 70)

bge_classifier = LinearSVC(
    C=1.0,
    random_state=42
)

bge_classifier.fit(
    X_train,
    y_train
)

print("Training Complete.")

# =====================================================
# PREDICTIONS
# =====================================================

print("\nRunning predictions...")

bge_predictions = bge_classifier.predict(
    X_test
)

# =====================================================
# METRICS
# =====================================================

bge_accuracy = accuracy_score(
    y_test,
    bge_predictions
)

bge_precision = precision_score(
    y_test,
    bge_predictions,
    average="weighted",
    zero_division=0
)

bge_recall = recall_score(
    y_test,
    bge_predictions,
    average="weighted",
    zero_division=0
)

bge_f1 = f1_score(
    y_test,
    bge_predictions,
    average="weighted",
    zero_division=0
)

# =====================================================
# RESULTS
# =====================================================

print("\n")
print("=" * 70)
print("BGE RESULTS")
print("=" * 70)

print(f"Accuracy      : {bge_accuracy:.4f}")
print(f"Precision     : {bge_precision:.4f}")
print(f"Recall        : {bge_recall:.4f}")
print(f"Weighted F1   : {bge_f1:.4f}")

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        bge_predictions,
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
        bge_predictions
    )
)

# =====================================================
# STORE RESULTS
# (Used in Part 2)
# =====================================================

results = []

results.append({
    "Method": "BGE Embeddings",
    "Accuracy": bge_accuracy,
    "Precision": bge_precision,
    "Recall": bge_recall,
    "Weighted F1": bge_f1
})

print("\nBGE evaluation complete.")

print("\nContinue with Part 2 to evaluate TF-IDF and compare both methods.")

# =====================================================
# PART 2
# TF-IDF BASELINE
# =====================================================

from sklearn.feature_extraction.text import TfidfVectorizer

print("\n")
print("=" * 70)
print("TF-IDF BASELINE")
print("=" * 70)

# -----------------------------------------------------
# BUILD TF-IDF FEATURES
# -----------------------------------------------------

texts = training_df["Issue_Only"].fillna("").astype(str)

tfidf = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    max_features=10000,
    ngram_range=(1, 2)
)

X_tfidf = tfidf.fit_transform(texts)

print(f"TF-IDF Shape : {X_tfidf.shape}")

# -----------------------------------------------------
# TRAIN / TEST SPLIT
# -----------------------------------------------------

(
    X_train,
    X_test,
    y_train,
    y_test
) = train_test_split(
    X_tfidf,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

# -----------------------------------------------------
# TRAIN CLASSIFIER
# -----------------------------------------------------

print("\nTraining TF-IDF + LinearSVC...")

tfidf_classifier = LinearSVC(
    C=1.0,
    random_state=42
)

tfidf_classifier.fit(
    X_train,
    y_train
)

print("Training Complete.")

# -----------------------------------------------------
# PREDICTIONS
# -----------------------------------------------------

tfidf_predictions = tfidf_classifier.predict(
    X_test
)

# -----------------------------------------------------
# METRICS
# -----------------------------------------------------

tfidf_accuracy = accuracy_score(
    y_test,
    tfidf_predictions
)

tfidf_precision = precision_score(
    y_test,
    tfidf_predictions,
    average="weighted",
    zero_division=0
)

tfidf_recall = recall_score(
    y_test,
    tfidf_predictions,
    average="weighted",
    zero_division=0
)

tfidf_f1 = f1_score(
    y_test,
    tfidf_predictions,
    average="weighted",
    zero_division=0
)

print("\n")
print("=" * 70)
print("TF-IDF RESULTS")
print("=" * 70)

print(f"Accuracy      : {tfidf_accuracy:.4f}")
print(f"Precision     : {tfidf_precision:.4f}")
print(f"Recall        : {tfidf_recall:.4f}")
print(f"Weighted F1   : {tfidf_f1:.4f}")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        tfidf_predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        tfidf_predictions
    )
)

# -----------------------------------------------------
# SAVE RESULTS
# -----------------------------------------------------

results.append({
    "Method": "TF-IDF",
    "Accuracy": tfidf_accuracy,
    "Precision": tfidf_precision,
    "Recall": tfidf_recall,
    "Weighted F1": tfidf_f1
})

# =====================================================
# COMPARISON
# =====================================================

comparison = pd.DataFrame(results)

comparison = comparison.round(4)

print("\n")
print("=" * 70)
print("BGE vs TF-IDF")
print("=" * 70)

print(comparison.to_string(index=False))

comparison.to_csv(
    "embedding_comparison_results.csv",
    index=False
)

# =====================================================
# WINNER
# =====================================================

print("\n")
print("=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

winner = comparison.loc[
    comparison["Weighted F1"].idxmax()
]

print(f"Winner : {winner['Method']}")

print(f"\nAccuracy Improvement : {(bge_accuracy - tfidf_accuracy):.4f}")

print(f"Weighted F1 Improvement : {(bge_f1 - tfidf_f1):.4f}")

print("\nSaved:")
print("embedding_comparison_results.csv")

print("\nDone.")
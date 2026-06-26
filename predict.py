"""
predict.py
==========

PART 1

Loads the trained models and predicts

1. Category
2. Priority

Part 2:
    - Compare with Human Priority
    - Decision Mapping
    - Pretty Output
"""

import joblib
from sentence_transformers import SentenceTransformer

# =====================================================
# LOAD MODELS
# =====================================================

print("=" * 70)
print("LOADING MODELS")
print("=" * 70)

print("\nLoading Category Classifier...")
category_classifier = joblib.load("linear_svc_ticket_classifier.pkl")
print("Done.")

print("\nLoading Priority Classifier...")
priority_classifier = joblib.load("priority_classifier.pkl")
print("Done.")

print("\nLoading Category Labels...")
category_labels = joblib.load("ticket_labels.pkl")
print(f"{len(category_labels)} labels loaded.")

print("\nLoading Priority Labels...")
priority_labels = joblib.load("priority_labels.pkl")
print(f"{len(priority_labels)} labels loaded.")

print("\nLoading BGE Embedding Model...")
embedding_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
print("Ready.")

# =====================================================
# PRIORITY DECISION ENGINE
# =====================================================

PRIORITY_DECISION = {
    ("Low",    "Low"):    "Confirmed",
    ("Low",    "Medium"): "Review Required",
    ("Low",    "High"):   "Critical Escalation",
    ("Medium", "Low"):    "Review Required",
    ("Medium", "Medium"): "Confirmed",
    ("Medium", "High"):   "Review Required",
    ("High",   "Low"):    "Priority Review Required",
    ("High",   "Medium"): "Priority Review Required",
    ("High",   "High"):   "Confirmed",
}

# =====================================================
# PREDICTION LOOP
# =====================================================

while True:

    print("\n" + "=" * 70)
    print("CUSTOMER SUPPORT TICKET")
    print("=" * 70)

    company = input("\nCompany (or 'quit'): ").strip()

    if company.lower() in ["quit", "exit", "q"]:
        break

    ticket = input("\nTicket Description: ").strip()

    human_priority = input(
        "\nHuman Assigned Priority (Low/Medium/High): "
    ).strip().title()

    if human_priority not in ["Low", "Medium", "High"]:
        print("\nInvalid Priority.")
        continue

    # =====================================================
    # GENERATE EMBEDDING
    # =====================================================

    embedding = embedding_model.encode([ticket])

    # =====================================================
    # CATEGORY PREDICTION
    # =====================================================

    predicted_category = category_classifier.predict(embedding)[0]

    # =====================================================
    # PRIORITY PREDICTION
    # =====================================================

    predicted_priority = priority_classifier.predict(embedding)[0]

    # =====================================================
    # PART 2 — DECISION MAPPING
    # =====================================================

    decision = PRIORITY_DECISION.get(
        (human_priority, predicted_priority),
        "Review Required"
    )

    # =====================================================
    # OPTIONAL CONFIDENCE SCORES
    # =====================================================

    try:
        category_score = (
            category_classifier.decision_function(embedding).max()
        )
    except Exception:
        category_score = None

    try:
        priority_score = (
            priority_classifier.decision_function(embedding).max()
        )
    except Exception:
        priority_score = None

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    print("\n")
    print("=" * 70)
    print("CUSTOMER SUPPORT TICKET")
    print("=" * 70)

    print(f"\nCompany")
    print(company)

    print(f"\nTicket")
    print(ticket)

    print(f"\nPredicted Category")
    print(predicted_category)

    print(f"\nHuman Priority")
    print(human_priority)

    print(f"\nPredicted Priority")
    print(predicted_priority)

    print(f"\nDecision")
    print(decision)

    print("\n" + "-" * 42)

    print("\nCategory Confidence")
    print(round(float(category_score), 3) if category_score is not None else "N/A")

    print("\nPriority Confidence")
    print(round(float(priority_score), 3) if priority_score is not None else "N/A")

    print("=" * 70)

print("\nProgram Closed.")
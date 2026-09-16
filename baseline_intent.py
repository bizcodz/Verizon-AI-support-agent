import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# Configuration
# ============================================================

INPUT_FILE = "golden_set.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ============================================================
# Load data
# ============================================================

df = pd.read_csv(INPUT_FILE)

required_columns = [
    "customer_text",
    "final_intent"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df = df.dropna(subset=required_columns).copy()

df["customer_text"] = df["customer_text"].astype(str)
df["final_intent"] = df["final_intent"].astype(str)


print("=" * 70)
print("HIVER - INTENT CLASSIFICATION BASELINE")
print("=" * 70)

print(f"Dataset size: {len(df)}")
print("\nIntent distribution:")
print(df["final_intent"].value_counts().sort_index())


# ============================================================
# Train / test split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    df["customer_text"],
    df["final_intent"],
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df["final_intent"]
)

print("\nTrain size:", len(X_train))
print("Test size :", len(X_test))


# ============================================================
# Baseline model
#
# TF-IDF:
#   converts text into weighted word/phrase features
#
# Logistic Regression:
#   learns which features correspond to each intent
# ============================================================

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        )
    )
])


# ============================================================
# Train
# ============================================================

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training complete.")


# ============================================================
# Predict
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

macro_f1 = f1_score(
    y_test,
    y_pred,
    average="macro"
)

weighted_f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)


print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"Accuracy    : {accuracy:.4f}")
print(f"Macro F1    : {macro_f1:.4f}")
print(f"Weighted F1 : {weighted_f1:.4f}")


# ============================================================
# Per-intent report
# ============================================================

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# Confusion matrix
# ============================================================

labels = sorted(df["final_intent"].unique())

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\nConfusion Matrix:")
print(cm_df)


# ============================================================
# Save predictions
# ============================================================

results = pd.DataFrame({
    "customer_text": X_test.values,
    "gold_intent": y_test.values,
    "predicted_intent": y_pred
})

results["correct"] = (
    results["gold_intent"] ==
    results["predicted_intent"]
)

results.to_csv(
    "baseline_predictions.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:")
print("  baseline_predictions.csv")


# ============================================================
# Show errors
# ============================================================

errors = results[
    results["gold_intent"] != results["predicted_intent"]
].copy()

print("\nNumber of errors:", len(errors))

print("\nSample errors:")

for _, row in errors.head(15).iterrows():

    print("\n---")
    print("TEXT:", row["customer_text"])
    print("GOLD:", row["gold_intent"])
    print("PRED:", row["predicted_intent"])


print("\nDone.")
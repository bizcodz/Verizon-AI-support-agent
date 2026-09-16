import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from agent_v1 import (
    load_retriever,
    run_agent,
    INTENTS,
)


GOLDEN_PATH = "golden_set.csv"
OUTPUT_PATH = "agent_v1_predictions.csv"


print("=" * 70)
print("AGENT V1 — GOLDEN SET EVALUATION")
print("=" * 70)


# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("\nLoading golden set...")

df = pd.read_csv(GOLDEN_PATH)

print("Golden examples:", len(df))
print("Columns:", list(df.columns))


# Detect intent column
if "final_intent" in df.columns:
    gold_column = "final_intent"
elif "gold_intent" in df.columns:
    gold_column = "gold_intent"
elif "intent" in df.columns:
    gold_column = "intent"
else:
    raise ValueError(
        "Could not find gold intent column. "
        "Expected final_intent, gold_intent, or intent."
    )


if "customer_text" not in df.columns:
    raise ValueError("customer_text column not found.")


# ============================================================
# LOAD AGENT
# ============================================================

print("\nLoading retriever...")

retriever = load_retriever()

print("Running Agent V1 on all golden cases...\n")


# ============================================================
# RUN AGENT
# ============================================================

rows = []

for i, row in df.iterrows():

    customer_text = str(row["customer_text"])
    gold_intent = str(row[gold_column])

    result = run_agent(
        customer_text,
        retriever,
    )

    evidence = result["evidence"]

    best_similarity = (
        evidence[0]["similarity"]
        if evidence
        else 0.0
    )

    best_response_type = (
        evidence[0]["response_type"]
        if evidence
        else "none"
    )

    best_historical_customer = (
        evidence[0]["historical_customer"]
        if evidence
        else ""
    )

    best_historical_reply = (
        evidence[0]["historical_reply"]
        if evidence
        else ""
    )

    top3_max_similarity = (
        max(x["similarity"] for x in evidence)
        if evidence
        else 0.0
    )

    rows.append({
        "case_id": row.get(
            "case_id",
            f"CASE-{i + 1:03d}"
        ),
        "customer_text": customer_text,
        "gold_intent": gold_intent,
        "predicted_intent": result["intent"],
        "intent_correct": (
            gold_intent == result["intent"]
        ),
        "should_escalate": result["should_escalate"],
        "decision": (
            "ESCALATE"
            if result["should_escalate"]
            else "AUTO_HANDLE"
        ),
        "escalation_reason": result["escalation_reason"],
        "reply": result["reply"],
        "evidence_count": len(evidence),
        "best_similarity": best_similarity,
        "top3_max_similarity": top3_max_similarity,
        "best_response_type": best_response_type,
        "best_historical_customer": best_historical_customer,
        "best_historical_reply": best_historical_reply,
    })


results = pd.DataFrame(rows)


# ============================================================
# INTENT METRICS
# ============================================================

y_true = results["gold_intent"]
y_pred = results["predicted_intent"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


print("=" * 70)
print("INTENT CLASSIFICATION")
print("=" * 70)

print(f"\nAccuracy:    {accuracy:.4f}")
print(f"Macro F1:    {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")


labels = list(INTENTS.keys())

print("\nClassification report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
)


# ============================================================
# DECISION METRICS
# ============================================================

total = len(results)

escalated = (
    results["should_escalate"] == True
).sum()

auto_handled = (
    results["should_escalate"] == False
).sum()


print("=" * 70)
print("AUTO-HANDLE / ESCALATION")
print("=" * 70)

print(
    f"\nAuto-handled: {auto_handled}/{total} "
    f"({auto_handled / total:.1%})"
)

print(
    f"Escalated:    {escalated}/{total} "
    f"({escalated / total:.1%})"
)


# ============================================================
# CLASSIFICATION × DECISION
# ============================================================

correct_auto = (
    (results["intent_correct"] == True)
    &
    (results["should_escalate"] == False)
).sum()

wrong_auto = (
    (results["intent_correct"] == False)
    &
    (results["should_escalate"] == False)
).sum()

correct_escalated = (
    (results["intent_correct"] == True)
    &
    (results["should_escalate"] == True)
).sum()

wrong_escalated = (
    (results["intent_correct"] == False)
    &
    (results["should_escalate"] == True)
).sum()


print("\n" + "=" * 70)
print("CLASSIFICATION × DECISION")
print("=" * 70)

print(
    f"\nCorrect intent + AUTO_HANDLE : "
    f"{correct_auto}"
)

print(
    f"Wrong intent   + AUTO_HANDLE : "
    f"{wrong_auto}"
)

print(
    f"Correct intent + ESCALATE    : "
    f"{correct_escalated}"
)

print(
    f"Wrong intent   + ESCALATE    : "
    f"{wrong_escalated}"
)


# ============================================================
# AUTO-HANDLE PRECISION
# ============================================================

if auto_handled > 0:

    auto_intent_precision = (
        correct_auto / auto_handled
    )

else:

    auto_intent_precision = 0.0


print("\n" + "=" * 70)
print("AUTO-HANDLE SAFETY PROXY")
print("=" * 70)

print(
    "\nAmong AUTO_HANDLE cases:"
)

print(
    f"Correct predicted intent: "
    f"{auto_intent_precision:.1%}"
)

print(
    f"Wrong predicted intent:   "
    f"{1 - auto_intent_precision:.1%}"
)


# ============================================================
# RETRIEVAL
# ============================================================

with_evidence = (
    results["evidence_count"] > 0
).sum()

similarity_020 = (
    results["best_similarity"] >= 0.20
).sum()

similarity_030 = (
    results["best_similarity"] >= 0.30
).sum()

similarity_040 = (
    results["best_similarity"] >= 0.40
).sum()


print("\n" + "=" * 70)
print("RETRIEVAL COVERAGE")
print("=" * 70)

print(
    f"\nCases with evidence: "
    f"{with_evidence}/{total} "
    f"({with_evidence / total:.1%})"
)

print(
    f"Best similarity >= 0.20: "
    f"{similarity_020}/{total} "
    f"({similarity_020 / total:.1%})"
)

print(
    f"Best similarity >= 0.30: "
    f"{similarity_030}/{total} "
    f"({similarity_030 / total:.1%})"
)

print(
    f"Best similarity >= 0.40: "
    f"{similarity_040}/{total} "
    f"({similarity_040 / total:.1%})"
)


# ============================================================
# RESPONSE TYPES
# ============================================================

print("\n" + "=" * 70)
print("TOP EVIDENCE RESPONSE TYPES")
print("=" * 70)

print(
    results[
        "best_response_type"
    ].value_counts(
        dropna=False
    )
)


# ============================================================
# INTENT ERRORS
# ============================================================

errors = results[
    results["intent_correct"] == False
].copy()


print("\n" + "=" * 70)
print("INTENT ERRORS")
print("=" * 70)

print(
    f"\nTotal errors: "
    f"{len(errors)}/{total}"
)


if len(errors) > 0:

    display_columns = [
        "case_id",
        "gold_intent",
        "predicted_intent",
        "decision",
        "best_similarity",
        "customer_text",
    ]

    print(
        "\nSample errors:\n"
    )

    print(
        errors[
            display_columns
        ].head(20).to_string(
            index=False
        )
    )


# ============================================================
# RISKY AUTO-HANDLE CASES
# ============================================================

risky_auto = results[
    (results["should_escalate"] == False)
    &
    (results["intent_correct"] == False)
].copy()


print("\n" + "=" * 70)
print("RISKY AUTO-HANDLE CASES")
print("=" * 70)

print(
    f"\nWrong intent but AUTO_HANDLE: "
    f"{len(risky_auto)}"
)


if len(risky_auto) > 0:

    print(
        risky_auto[
            [
                "case_id",
                "gold_intent",
                "predicted_intent",
                "best_similarity",
                "best_response_type",
                "customer_text",
                "reply",
            ]
        ].head(20).to_string(
            index=False
        )
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=labels,
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels,
)

print("\nRows = GOLD")
print("Columns = PREDICTED\n")

print(cm_df.to_string())


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False,
)

print("\n" + "=" * 70)
print("OUTPUT")
print("=" * 70)

print(
    f"\nSaved predictions: "
    f"{OUTPUT_PATH}"
)

print("\nDone.")
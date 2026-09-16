import sys
from pathlib import Path

import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# pyrefly: ignore [missing-import]
from agent_v2_v24 import (
    assign_intent,
    retrieve_evidence,
    choose_grounding_evidence,
    escalation_decision,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

GOLDEN_FILE = ROOT / "data" / "golden_set.csv"
RETRIEVER_FILE = ROOT / "artifacts" / "verizon_resolution_retriever_v3.pkl"
OUTPUT_FILE = ROOT / "results" / "agent_v24_predictions.csv"


def main():
    print("=" * 70)
    print("AGENT V2.4 - FULL GOLDEN SET EVALUATION")
    print("=" * 70)

    df = pd.read_csv(GOLDEN_FILE)
    retriever = joblib.load(RETRIEVER_FILE)

    rows = []

    total = len(df)

    for i, row in df.iterrows():
        customer_text = str(row["customer_text"])
        gold_intent = str(row["final_intent"])

        predicted_intent = assign_intent(customer_text)

        evidence = retrieve_evidence(
            customer_text,
            predicted_intent,
            retriever,
        )

        selected_evidence = choose_grounding_evidence(
            evidence,
            customer_text,
            predicted_intent,
        )

        should_escalate, escalation_reason = escalation_decision(
            customer_text,
            predicted_intent,
            evidence,
        )
        rows.append({
            "case_id": row["case_id"],
            "customer_text": customer_text,
            "gold_intent": gold_intent,
            "predicted_intent": predicted_intent,
            "intent_correct": predicted_intent == gold_intent,
            "should_escalate": should_escalate,
            "escalation_reason": escalation_reason,
            "evidence_available": len(evidence) > 0,
            "evidence_selected": selected_evidence is not None,
            "selected_similarity": (
                selected_evidence["similarity"]
                if selected_evidence
                else 0.0
            ),
            "selected_response_type": (
                selected_evidence.get("response_type", "")
                if selected_evidence
                else ""
            ),
        })

        if (i + 1) % 25 == 0 or i + 1 == total:
            print(f"Processed {i + 1}/{total}")

    results = pd.DataFrame(rows)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    y_true = results["gold_intent"]
    y_pred = results["predicted_intent"]

    print()
    print("Accuracy:", f"{accuracy_score(y_true, y_pred):.4f}")
    print("Macro F1:", f"{f1_score(y_true, y_pred, average='macro'):.4f}")
    print("Weighted F1:", f"{f1_score(y_true, y_pred, average='weighted'):.4f}")

    print()
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()



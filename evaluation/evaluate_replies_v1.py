import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reply_generator_v2 import run_agent

GOLDEN_FILE = ROOT / "data" / "golden_set.csv"
RETRIEVER_FILE = ROOT / "artifacts" / "verizon_resolution_retriever_v3.pkl"
OUTPUT_FILE = ROOT / "results" / "reply_baseline_v1_predictions.csv"


def main():
    print("=" * 70)
    print("REPLY BASELINE V1 - FULL GOLDEN SET EVALUATION")
    print("=" * 70)

    df = pd.read_csv(GOLDEN_FILE)
    retriever = joblib.load(RETRIEVER_FILE)

    rows = []
    total = len(df)

    for i, row in df.iterrows():
        customer_text = str(row["customer_text"])
        gold_intent = str(row["final_intent"])

        result = run_agent(customer_text, retriever)

        evidence = result.get("selected_evidence")

        rows.append({
            "case_id": row["case_id"],
            "customer_text": customer_text,
            "gold_intent": gold_intent,
            "predicted_intent": result["predicted_intent"],
            "intent_correct": result["predicted_intent"] == gold_intent,
            "should_escalate": result["should_escalate"],
            "escalation_reason": result["escalation_reason"],
            "reply": result["reply"],
            "historical_customer": (
                evidence["historical_customer"]
                if evidence
                else ""
            ),
            "historical_reply": (
                evidence["historical_reply"]
                if evidence
                else ""
            ),
            "similarity": (
                evidence["similarity"]
                if evidence
                else 0.0
            ),
            "response_type": (
                evidence.get("response_type", "")
                if evidence
                else ""
            ),
            "evidence_selected": evidence is not None,
        })

        if (i + 1) % 25 == 0 or i + 1 == total:
            print(f"Processed {i + 1}/{total}")

    results = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Replies generated: {len(results)}/{total}")
    print(
        "Intent-correct:",
        f"{results['intent_correct'].sum()}/{total}"
    )
    print(
        "Escalated:",
        f"{results['should_escalate'].sum()}/{total}"
    )
    print(
        "Evidence selected:",
        f"{results['evidence_selected'].sum()}/{total}"
    )

    selected = results[results["evidence_selected"]]

    if len(selected) > 0:
        print(
            "Mean selected similarity:",
            f"{selected['similarity'].mean():.4f}"
        )
        print(
            "Median selected similarity:",
            f"{selected['similarity'].median():.4f}"
        )

    print()
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

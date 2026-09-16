import joblib
import pandas as pd

from reply_generator_v2 import run_agent


GOLDEN_FILE = "golden_set.csv"
RETRIEVER_FILE = "verizon_resolution_retriever_v3.pkl"
OUTPUT_FILE = "reply_baseline_v1_predictions.csv"


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

        result = run_agent(
            customer_text,
            retriever,
        )

        selected = result["selected_evidence"]

        if selected is None:

            evidence_customer = ""
            evidence_reply = ""
            evidence_similarity = None
            evidence_response_type = ""

        else:

            evidence_customer = selected[
                "historical_customer"
            ]

            evidence_reply = selected[
                "historical_reply"
            ]

            evidence_similarity = selected[
                "similarity"
            ]

            evidence_response_type = selected[
                "response_type"
            ]

        rows.append({

            "case_id":
                row["case_id"],

            "customer_text":
                customer_text,

            "gold_intent":
                gold_intent,

            "predicted_intent":
                result["predicted_intent"],

            "intent_correct":
                result["predicted_intent"] == gold_intent,

            "should_escalate":
                result["should_escalate"],

            "escalation_reason":
                result["escalation_reason"],

            "reply":
                result["reply"],

            "selected_evidence_customer":
                evidence_customer,

            "selected_evidence_reply":
                evidence_reply,

            "selected_evidence_similarity":
                evidence_similarity,

            "selected_evidence_response_type":
                evidence_response_type,

            "evidence_available":
                result["evidence_count"] > 0,

            "evidence_selected":
                selected is not None,
        })

        if (i + 1) % 25 == 0:
            print(
                f"Processed {i + 1}/{total} cases..."
            )

    results = pd.DataFrame(rows)

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)

    print(
        "Total cases:",
        len(results)
    )

    print(
        "Replies generated:",
        results["reply"].notna().sum()
    )

    print(
        "Evidence available:",
        f"{results['evidence_available'].sum()}/"
        f"{len(results)}"
    )

    print(
        "Evidence selected:",
        f"{results['evidence_selected'].sum()}/"
        f"{len(results)}"
    )

    print(
        "Intent-correct cases:",
        f"{results['intent_correct'].sum()}/"
        f"{len(results)}"
    )

    print()
    print("=" * 70)
    print("REPLY / ESCALATION BREAKDOWN")
    print("=" * 70)

    print(
        results["should_escalate"]
        .value_counts()
        .rename({
            True: "Escalated",
            False: "Auto-handled",
        })
    )

    print()
    print("=" * 70)
    print("SELECTED RESPONSE TYPES")
    print("=" * 70)

    print(
        results[
            "selected_evidence_response_type"
        ].value_counts(dropna=False)
    )

    print()
    print("=" * 70)
    print("SIMILARITY")
    print("=" * 70)

    similarity = results[
        "selected_evidence_similarity"
    ].dropna()

    if len(similarity) > 0:

        print(
            "Mean:",
            f"{similarity.mean():.4f}"
        )

        print(
            "Median:",
            f"{similarity.median():.4f}"
        )

        print(
            ">= 0.20:",
            f"{(similarity >= 0.20).sum()}/"
            f"{len(similarity)}"
        )

        print(
            ">= 0.30:",
            f"{(similarity >= 0.30).sum()}/"
            f"{len(similarity)}"
        )

        print(
            ">= 0.40:",
            f"{(similarity >= 0.40).sum()}/"
            f"{len(similarity)}"
        )

        print(
            ">= 0.50:",
            f"{(similarity >= 0.50).sum()}/"
            f"{len(similarity)}"
        )

    print()
    print(
        f"Saved predictions to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
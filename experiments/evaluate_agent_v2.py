import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from agent_v2 import (
    assign_intent,
    retrieve_evidence,
    choose_grounding_evidence,
    escalation_decision,
)


GOLDEN_FILE = "golden_set.csv"
RETRIEVER_FILE = "verizon_resolution_retriever_v3.pkl"
OUTPUT_FILE = "agent_v2_predictions.csv"


def main():

    print("=" * 70)
    print("AGENT V2.2.1 - FULL GOLDEN SET EVALUATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load golden set and retriever
    # ------------------------------------------------------------

    df = pd.read_csv(GOLDEN_FILE)
    retriever = joblib.load(RETRIEVER_FILE)

    required_columns = [
        "case_id",
        "customer_text",
        "final_intent",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in golden_set.csv: {missing}"
        )

    print(f"Golden cases: {len(df)}")

    predictions = []

    # ------------------------------------------------------------
    # Run agent on every golden case
    # ------------------------------------------------------------

    for idx, row in df.iterrows():

        case_id = row["case_id"]
        customer_text = str(row["customer_text"])
        gold_intent = str(row["final_intent"]).strip()

        # --------------------------------------------------------
        # 1. Intent classification
        # --------------------------------------------------------

        predicted_intent = assign_intent(
            customer_text
        )

        # --------------------------------------------------------
        # 2. Retrieve historical evidence
        # --------------------------------------------------------

        evidence = retrieve_evidence(
            customer_text,
            predicted_intent,
            retriever,
        )

        # --------------------------------------------------------
        # 3. Select grounding evidence for evaluation
        #
        # This is independent measurement of the grounding gate.
        # --------------------------------------------------------

        selected_evidence = choose_grounding_evidence(
            evidence,
            customer_text,
            predicted_intent,
        )

        # --------------------------------------------------------
        # 4. Escalation policy
        #
        # IMPORTANT:
        # escalation_decision() expects the FULL evidence list
        # and performs its own grounding check internally.
        #
        # It returns:
        #
        #     (should_escalate, reason)
        # --------------------------------------------------------

        should_escalate, escalation_reason = (
            escalation_decision(
                customer_text,
                predicted_intent,
                evidence,
            )
        )

        predicted_escalate = (
            "YES"
            if should_escalate
            else "NO"
        )

        # --------------------------------------------------------
        # 5. Evidence information
        # --------------------------------------------------------

        evidence_available = (
            len(evidence) > 0
        )

        if selected_evidence is not None:

            evidence_passed_gate = True

            selected_similarity = float(
                selected_evidence.get(
                    "similarity",
                    0.0,
                )
            )

            selected_response_type = (
                selected_evidence.get(
                    "response_type",
                    "unknown",
                )
            )

            selected_historical_customer = (
                selected_evidence.get(
                    "historical_customer",
                    "",
                )
            )

            selected_historical_reply = (
                selected_evidence.get(
                    "historical_reply",
                    "",
                )
            )

        else:

            evidence_passed_gate = False

            selected_similarity = 0.0

            selected_response_type = ""

            selected_historical_customer = ""

            selected_historical_reply = ""

        # --------------------------------------------------------
        # 6. Intent correctness
        # --------------------------------------------------------

        intent_correct = (
            predicted_intent == gold_intent
        )

        # --------------------------------------------------------
        # 7. Safety proxy
        #
        # No human gold escalation labels currently exist.
        #
        # Therefore:
        #
        # Correct intent + AUTO = correct-intent auto-handle
        # Wrong intent + AUTO   = risky auto-handle
        # --------------------------------------------------------

        correct_auto_handle = (
            not should_escalate
            and intent_correct
        )

        risky_auto_handle = (
            not should_escalate
            and not intent_correct
        )

        # --------------------------------------------------------
        # 8. Save result
        # --------------------------------------------------------

        predictions.append({

            "case_id":
                case_id,

            "customer_text":
                customer_text,

            "gold_intent":
                gold_intent,

            "predicted_intent":
                predicted_intent,

            "intent_correct":
                intent_correct,

            "predicted_should_escalate":
                predicted_escalate,

            "escalation_reason":
                escalation_reason,

            "evidence_available":
                evidence_available,

            "evidence_passed_consistency_gate":
                evidence_passed_gate,

            "selected_similarity":
                selected_similarity,

            "selected_response_type":
                selected_response_type,

            "selected_historical_customer":
                selected_historical_customer,

            "selected_historical_reply":
                selected_historical_reply,

            "correct_auto_handle":
                correct_auto_handle,

            "risky_wrong_intent_auto_handle":
                risky_auto_handle,
        })

        # --------------------------------------------------------
        # Progress
        # --------------------------------------------------------

        if (idx + 1) % 25 == 0:

            print(
                f"Processed {idx + 1}/{len(df)} cases..."
            )

    # ------------------------------------------------------------
    # Results dataframe
    # ------------------------------------------------------------

    results = pd.DataFrame(
        predictions
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
    )

    # ------------------------------------------------------------
    # Intent metrics
    # ------------------------------------------------------------

    y_true = results[
        "gold_intent"
    ]

    y_pred = results[
        "predicted_intent"
    ]

    accuracy = accuracy_score(
        y_true,
        y_pred,
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

    total = len(results)

    # ------------------------------------------------------------
    # Auto-handle metrics
    # ------------------------------------------------------------

    auto_handles = (
        results[
            "predicted_should_escalate"
        ] == "NO"
    ).sum()

    escalations = (
        results[
            "predicted_should_escalate"
        ] == "YES"
    ).sum()

    correct_auto = (
        results[
            "correct_auto_handle"
        ]
    ).sum()

    risky_auto = (
        results[
            "risky_wrong_intent_auto_handle"
        ]
    ).sum()

    auto_handle_rate = (
        auto_handles / total
    )

    escalation_rate = (
        escalations / total
    )

    if auto_handles > 0:

        auto_handle_safety = (
            correct_auto /
            auto_handles
        )

    else:

        auto_handle_safety = 0.0

    # ------------------------------------------------------------
    # Evidence metrics
    # ------------------------------------------------------------

    evidence_available = (
        results[
            "evidence_available"
        ]
    ).sum()

    evidence_passed = (
        results[
            "evidence_passed_consistency_gate"
        ]
    ).sum()

    # ------------------------------------------------------------
    # HEADLINE RESULTS
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("HEADLINE RESULTS")
    print("=" * 70)

    print(
        f"Total cases:                    "
        f"{total}"
    )

    print(
        f"Intent accuracy:                "
        f"{accuracy:.4f} "
        f"({accuracy:.1%})"
    )

    print(
        f"Macro F1:                       "
        f"{macro_f1:.4f} "
        f"({macro_f1:.1%})"
    )

    print(
        f"Weighted F1:                    "
        f"{weighted_f1:.4f} "
        f"({weighted_f1:.1%})"
    )

    # ------------------------------------------------------------
    # AUTO-HANDLE / ESCALATION
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("AUTO-HANDLE / ESCALATION")
    print("=" * 70)

    print(
        f"Auto-handled:                   "
        f"{auto_handles}/{total} "
        f"({auto_handle_rate:.1%})"
    )

    print(
        f"Escalated:                      "
        f"{escalations}/{total} "
        f"({escalation_rate:.1%})"
    )

    print(
        "Escalation decision agreement:  "
        "Not evaluated "
        "(no human escalation labels)"
    )

    # ------------------------------------------------------------
    # SAFETY PROXY
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("AUTO-HANDLE SAFETY PROXY")
    print("=" * 70)

    print(
        f"Correct-intent auto-handles:    "
        f"{correct_auto}"
    )

    print(
        f"Wrong-intent auto-handles:      "
        f"{risky_auto}"
    )

    print(
        f"Auto-handle safety proxy:       "
        f"{auto_handle_safety:.1%}"
    )

    # ------------------------------------------------------------
    # EVIDENCE
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("EVIDENCE")
    print("=" * 70)

    print(
        f"Evidence available:             "
        f"{evidence_available}/{total} "
        f"({evidence_available / total:.1%})"
    )

    print(
        f"Evidence passed consistency:     "
        f"{evidence_passed}/{total} "
        f"({evidence_passed / total:.1%})"
    )

    # ------------------------------------------------------------
    # SIMILARITY
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTED EVIDENCE SIMILARITY")
    print("=" * 70)

    similarities = results[
        "selected_similarity"
    ]

    print(
        f">= 0.20:                        "
        f"{(similarities >= 0.20).sum()}/{total}"
    )

    print(
        f">= 0.30:                        "
        f"{(similarities >= 0.30).sum()}/{total}"
    )

    print(
        f">= 0.40:                        "
        f"{(similarities >= 0.40).sum()}/{total}"
    )

    print(
        f">= 0.50:                        "
        f"{(similarities >= 0.50).sum()}/{total}"
    )

    print(
        f"Mean similarity:                "
        f"{similarities.mean():.4f}"
    )

    print(
        f"Median similarity:              "
        f"{similarities.median():.4f}"
    )

    # ------------------------------------------------------------
    # RESPONSE TYPES
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("SELECTED RESPONSE TYPES")
    print("=" * 70)

    print(
        results[
            "selected_response_type"
        ].value_counts(
            dropna=False
        )
    )

    # ------------------------------------------------------------
    # CLASSIFICATION REPORT
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
        )
    )

    # ------------------------------------------------------------
    # CONFUSION MATRIX
    # ------------------------------------------------------------

    labels = sorted(
        y_true.unique()
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"TRUE_{label}"
            for label in labels
        ],
        columns=[
            f"PRED_{label}"
            for label in labels
        ],
    )

    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)

    print(
        cm_df.to_string()
    )

    # ------------------------------------------------------------
    # RISKY AUTO-HANDLES
    # ------------------------------------------------------------

    risky = results[
        results[
            "risky_wrong_intent_auto_handle"
        ]
    ].copy()

    print()
    print("=" * 70)
    print(
        "RISKY WRONG-INTENT AUTO-HANDLES"
    )
    print("=" * 70)

    if len(risky) == 0:

        print("None.")

    else:

        print(
            f"Count: {len(risky)}"
        )

        for _, row in risky.iterrows():

            print()
            print("-" * 70)

            print(
                f"Case:       "
                f"{row['case_id']}"
            )

            print(
                f"Customer:   "
                f"{row['customer_text']}"
            )

            print(
                f"Gold:       "
                f"{row['gold_intent']}"
            )

            print(
                f"Predicted:  "
                f"{row['predicted_intent']}"
            )

            print(
                f"Similarity: "
                f"{row['selected_similarity']:.3f}"
            )

            print(
                f"Response:   "
                f"{row['selected_response_type']}"
            )

            print(
                f"Evidence:   "
                f"{row['selected_historical_customer']}"
            )

            print(
                f"Reply:      "
                f"{row['selected_historical_reply']}"
            )

            print(
                f"Reason:     "
                f"{row['escalation_reason']}"
            )

    # ------------------------------------------------------------
    # INTENT ERRORS
    # ------------------------------------------------------------

    errors = results[
        ~results[
            "intent_correct"
        ]
    ].copy()

    print()
    print("=" * 70)
    print(
        f"INTENT ERRORS ({len(errors)})"
    )
    print("=" * 70)

    for _, row in errors.iterrows():

        print(
            f"{row['case_id']} | "
            f"gold={row['gold_intent']} | "
            f"pred={row['predicted_intent']} | "
            f"{row['customer_text']}"
        )

    # ------------------------------------------------------------
    # FINAL
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"Saved predictions to: "
        f"{OUTPUT_FILE}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
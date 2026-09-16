import pandas as pd

df = pd.read_csv("reply_baseline_v1_predictions.csv")

auto = df[df["should_escalate"] == False].head(10)
escalated = df[df["should_escalate"] == True].head(10)
wrong = df[df["intent_correct"] == False].head(10)

selected = pd.concat([auto, escalated, wrong]).drop_duplicates("case_id")

remaining = df[~df["case_id"].isin(selected["case_id"])]

if len(selected) < 30:
    extra = remaining.sample(
        n=min(30 - len(selected), len(remaining)),
        random_state=42
    )
    selected = pd.concat([selected, extra]).drop_duplicates("case_id")

selected = selected.head(30)

cols = [
    "case_id",
    "customer_text",
    "gold_intent",
    "predicted_intent",
    "intent_correct",
    "should_escalate",
    "escalation_reason",
    "reply",
    "selected_evidence_customer",
    "selected_evidence_reply",
    "selected_evidence_similarity",
    "selected_evidence_response_type",
    "evidence_available",
    "evidence_selected"
]

selected[cols].to_csv("judge_calibration_30.csv", index=False)
selected[cols].to_excel("judge_calibration_30.xlsx", index=False)

print(f"Created {len(selected)} calibration cases")
print()
print("Case IDs:")
print(", ".join(selected["case_id"].astype(str)))
print()
print("Intent distribution:")
print(selected["gold_intent"].value_counts().sort_index())
print()
print("Escalation distribution:")
print(selected["should_escalate"].value_counts())
print()
print("Wrong intent cases:", (~selected["intent_correct"]).sum())
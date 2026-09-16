import pandas as pd

judge = pd.read_csv("llm_judge_30_results.csv")
calibration = pd.read_csv("judge_calibration_30.csv")

successful = judge[judge["relevance"].notna()]["case_id"]

df = calibration[calibration["case_id"].isin(successful)].copy()

df["human_relevance"] = ""
df["human_helpfulness"] = ""
df["human_grounding"] = ""
df["human_non_hallucination"] = ""
df["human_tone"] = ""
df["human_overall_acceptable"] = ""
df["human_escalation_appropriate"] = ""
df["human_reason"] = ""

cols = [
    "case_id",
    "customer_text",
    "predicted_intent",
    "should_escalate",
    "reply",
    "selected_evidence_customer",
    "selected_evidence_reply",
    "human_relevance",
    "human_helpfulness",
    "human_grounding",
    "human_non_hallucination",
    "human_tone",
    "human_overall_acceptable",
    "human_escalation_appropriate",
    "human_reason"
]

df[cols].to_excel("human_review_20.xlsx", index=False)

print(f"Created human review sheet with {len(df)} cases")
print("File: human_review_20.xlsx")
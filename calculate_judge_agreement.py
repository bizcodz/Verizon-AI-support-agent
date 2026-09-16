import pandas as pd
from sklearn.metrics import cohen_kappa_score

judge = pd.read_csv("llm_judge_30_results.csv")
human = pd.read_excel("human_review_20.xlsx")

df = judge.merge(human, on="case_id", how="inner")

judge_cols = [
    "relevance",
    "helpfulness",
    "grounding",
    "non_hallucination",
    "tone"
]

human_cols = [
    "human_relevance",
    "human_helpfulness",
    "human_grounding",
    "human_non_hallucination",
    "human_tone"
]

results = []

for j, h in zip(judge_cols, human_cols):
    tmp = df[[j, h]].dropna()

    judge_scores = tmp[j].astype(int)
    human_scores = tmp[h].astype(int)

    exact = (judge_scores == human_scores).mean()

    weighted_kappa = cohen_kappa_score(
        human_scores,
        judge_scores,
        weights="quadratic"
    )

    results.append({
        "dimension": j,
        "n": len(tmp),
        "exact_agreement": round(exact, 3),
        "quadratic_weighted_kappa": round(weighted_kappa, 3)
    })

binary = df[
    df["overall_acceptable"].notna() &
    df["human_overall_acceptable"].notna()
].copy()

binary["overall_acceptable"] = (
    binary["overall_acceptable"].astype(str).str.upper().str.strip()
)

binary["human_overall_acceptable"] = (
    binary["human_overall_acceptable"].astype(str).str.upper().str.strip()
)

overall_exact = (
    binary["overall_acceptable"] ==
    binary["human_overall_acceptable"]
).mean()

overall_kappa = cohen_kappa_score(
    binary["human_overall_acceptable"],
    binary["overall_acceptable"]
)

escalation = df[
    df["escalation_appropriate"].notna() &
    df["human_escalation_appropriate"].notna()
].copy()

escalation["escalation_appropriate"] = (
    escalation["escalation_appropriate"]
    .astype(str)
    .str.upper()
    .str.strip()
)

escalation["human_escalation_appropriate"] = (
    escalation["human_escalation_appropriate"]
    .astype(str)
    .str.upper()
    .str.strip()
)

escalation_exact = (
    escalation["escalation_appropriate"] ==
    escalation["human_escalation_appropriate"]
).mean()

escalation_kappa = cohen_kappa_score(
    escalation["human_escalation_appropriate"],
    escalation["escalation_appropriate"]
)

results_df = pd.DataFrame(results)

print()
print("JUDGE-HUMAN AGREEMENT")
print("=" * 60)
print(results_df.to_string(index=False))

print()
print("Overall acceptable")
print("-" * 60)
print("N:", len(binary))
print("Exact agreement:", round(overall_exact, 3))
print("Cohen kappa:", round(overall_kappa, 3))

print()
print("Escalation appropriateness")
print("-" * 60)
print("N:", len(escalation))
print("Exact agreement:", round(escalation_exact, 3))
print("Cohen kappa:", round(escalation_kappa, 3))

results_df.to_csv("judge_human_agreement.csv", index=False)

with open("judge_human_agreement_summary.txt", "w", encoding="utf-8") as f:
    f.write("JUDGE-HUMAN AGREEMENT\n")
    f.write("=" * 60 + "\n\n")
    f.write(results_df.to_string(index=False))
    f.write("\n\nOverall acceptable\n")
    f.write(f"N: {len(binary)}\n")
    f.write(f"Exact agreement: {overall_exact:.3f}\n")
    f.write(f"Cohen kappa: {overall_kappa:.3f}\n")
    f.write("\nEscalation appropriateness\n")
    f.write(f"N: {len(escalation)}\n")
    f.write(f"Exact agreement: {escalation_exact:.3f}\n")
    f.write(f"Cohen kappa: {escalation_kappa:.3f}\n")

print()
print("Saved:")
print("judge_human_agreement.csv")
print("judge_human_agreement_summary.txt")
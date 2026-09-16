import pandas as pd
import re

df = pd.read_csv("reply_baseline_v1_predictions.csv")

total = len(df)

reply_present = df["reply"].fillna("").str.strip().ne("")

internal_patterns = [
    r"historical case",
    r"retrieved",
    r"\bRAG\b",
    r"\binternal\b",
    r"system response",
    r"system generated",
    r"internal reasoning",
    r"model reasoning"
]

artifact_patterns = [
    r"check at\s+for",
    r"click here",
    r"https?://",
    r"www\.",
    r"@\w+",
    r"\^"
]

def has_pattern(text, patterns):
    text = str(text)
    return any(re.search(p, text, re.IGNORECASE) for p in patterns)

df["internal_leakage"] = df["reply"].fillna("").apply(
    lambda x: has_pattern(x, internal_patterns)
)

df["artifact_leakage"] = df["reply"].fillna("").apply(
    lambda x: has_pattern(x, artifact_patterns)
)

df["reply_length"] = df["reply"].fillna("").str.len()

selected_evidence = df["evidence_selected"].astype(str).str.upper() == "TRUE"

print("=" * 60)
print("AUTOMATED REPLY QUALITY CHECKS")
print("=" * 60)

print(f"Total cases: {total}")
print(f"Reply generated: {reply_present.sum()}/{total} ({reply_present.mean():.1%})")
print(
    f"Evidence available: "
    f"{df['evidence_available'].astype(str).str.upper().eq('TRUE').sum()}/{total}"
)
print(
    f"Evidence selected: "
    f"{selected_evidence.sum()}/{total} ({selected_evidence.mean():.1%})"
)

print()
print("Potential internal/meta leakage:")
print(
    f"{df['internal_leakage'].sum()}/{total} "
    f"({df['internal_leakage'].mean():.1%})"
)

print()
print("Potential artifact leakage:")
print(
    f"{df['artifact_leakage'].sum()}/{total} "
    f"({df['artifact_leakage'].mean():.1%})"
)

print()
print("Reply length:")
print(f"Mean: {df['reply_length'].mean():.1f} characters")
print(f"Median: {df['reply_length'].median():.1f} characters")

print()
print("Potentially problematic replies:")
problematic = df[
    (df["internal_leakage"]) |
    (df["artifact_leakage"])
]

if len(problematic):
    print(
        problematic[
            [
                "case_id",
                "gold_intent",
                "predicted_intent",
                "should_escalate",
                "reply"
            ]
        ].to_string(index=False)
    )
else:
    print("None detected.")

df.to_csv("reply_quality_checks.csv", index=False)

print()
print("Saved: reply_quality_checks.csv")
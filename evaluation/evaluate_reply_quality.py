from pathlib import Path
import pandas as pd
import re

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = ROOT / "results" / "reply_baseline_v1_predictions.csv"
OUTPUT_FILE = ROOT / "results" / "reply_quality_results.csv"

df = pd.read_csv(INPUT_FILE)

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

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("=" * 70)
print("REPLY QUALITY CHECK")
print("=" * 70)
print(f"Total cases: {total}")
print(f"Replies generated: {reply_present.sum()}/{total}")
print(
    f"Potential internal/meta leakage: "
    f"{df['internal_leakage'].sum()}/{total} "
    f"({df['internal_leakage'].mean() * 100:.1f}%)"
)
print(
    f"Potential artifact leakage: "
    f"{df['artifact_leakage'].sum()}/{total} "
    f"({df['artifact_leakage'].mean() * 100:.1f}%)"
)

lengths = df["reply"].fillna("").str.len()

print(f"Mean reply length: {lengths.mean():.1f} chars")
print(f"Median reply length: {lengths.median():.1f} chars")

print()
print(f"Saved: {OUTPUT_FILE}")

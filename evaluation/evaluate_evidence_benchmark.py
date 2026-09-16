import pandas as pd
import numpy as np


INPUT = "evidence_benchmark_review.xlsx"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("EVIDENCE QUALITY EVALUATION")
print("=" * 70)

df = pd.read_excel(
    INPUT,
    sheet_name="Evidence Review"
)

print("\nRows loaded:", len(df))


# ============================================================
# NORMALIZE ANNOTATIONS
# ============================================================

for column in [
    "relevant",
    "actionable",
    "safe_to_ground"
]:

    df[column] = (
        df[column]
        .astype(str)
        .str.strip()
        .str.upper()
    )


# ============================================================
# CHECK COMPLETENESS
# ============================================================

required = [
    "relevant",
    "actionable",
    "safe_to_ground"
]

print("\nAnnotation completeness:")

complete = True

for column in required:

    valid = df[column].isin(
        ["YES", "NO"]
    )

    missing = (~valid).sum()

    print(
        f"{column}:",
        len(df) - missing,
        "/",
        len(df),
        "valid"
    )

    if missing > 0:
        complete = False


if not complete:

    print(
        "\nERROR: Some annotations are missing "
        "or contain values other than YES/NO."
    )

    print(
        "\nPlease finish the annotations before "
        "running the evaluation."
    )

    raise SystemExit(1)


# ============================================================
# TOP-1 / TOP-3 METRICS
# ============================================================

# Each case has ranks 1, 2, 3.
# A Top-1 metric evaluates only rank 1.
# A Top-3 metric asks whether ANY of ranks 1–3
# contains a positive result.


def top1_rate(column):

    rank1 = df[
        df["rank"] == 1
    ]

    return (
        rank1[column] == "YES"
    ).mean()


def top3_rate(column):

    grouped = (
        df.groupby("case_id")[column]
        .apply(
            lambda x:
            (x == "YES").any()
        )
    )

    return grouped.mean()


# ============================================================
# CALCULATE
# ============================================================

metrics = {}

for column in [
    "relevant",
    "actionable",
    "safe_to_ground"
]:

    metrics[
        f"Top-1 {column}"
    ] = top1_rate(column)

    metrics[
        f"Top-3 {column}"
    ] = top3_rate(column)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

for name, value in metrics.items():

    print(
        f"{name:30s}: {value:.1%}"
    )


# ============================================================
# ADDITIONAL DIAGNOSTICS
# ============================================================

print("\n" + "=" * 70)
print("ANNOTATION COUNTS")
print("=" * 70)

for column in required:

    print(
        f"\n{column}:"
    )

    print(
        df[column]
        .value_counts()
    )


# ============================================================
# CASE-LEVEL SUMMARY
# ============================================================

case_summary = (
    df.groupby("case_id")
    .agg(
        gold_intent=("gold_intent", "first"),
        customer_text=("customer_text", "first"),

        top1_similarity=(
            "similarity",
            "first"
        ),

        top1_relevant=(
            "relevant",
            "first"
        ),

        top1_actionable=(
            "actionable",
            "first"
        ),

        top1_safe_to_ground=(
            "safe_to_ground",
            "first"
        )
    )
    .reset_index()
)


# ============================================================
# IDENTIFY CASES WHERE TOP-1 FAILED BUT TOP-3 SUCCEEDED
# ============================================================

top3 = (
    df.groupby("case_id")
    .agg(
        any_relevant=(
            "relevant",
            lambda x:
            (x == "YES").any()
        ),

        any_actionable=(
            "actionable",
            lambda x:
            (x == "YES").any()
        ),

        any_safe_to_ground=(
            "safe_to_ground",
            lambda x:
            (x == "YES").any()
        )
    )
    .reset_index()
)

case_summary = case_summary.merge(
    top3,
    on="case_id"
)


print("\n" + "=" * 70)
print("TOP-1 → TOP-3 RECOVERY")
print("=" * 70)

for metric in [
    "relevant",
    "actionable",
    "safe_to_ground"
]:

    top1 = (
        case_summary[
            f"top1_{metric}"
        ] == "YES"
    ).mean()

    top3_value = (
        case_summary[
            f"any_{metric}"
        ]
    ).mean()

    print(
        f"\n{metric}:"
    )

    print(
        f"  Top-1: {top1:.1%}"
    )

    print(
        f"  Top-3: {top3_value:.1%}"
    )

    print(
        f"  Gain:  {top3_value - top1:+.1%}"
    )


# ============================================================
# LOW-SIMILARITY FAILURES
# ============================================================

print("\n" + "=" * 70)
print("LOW-SIMILARITY CASES")
print("=" * 70)

low_similarity = df[
    (df["rank"] == 1)
    &
    (df["similarity"] < 0.20)
].copy()

print(
    "\nTop-1 cases with similarity < 0.20:",
    len(low_similarity)
)

if len(low_similarity) > 0:

    print(
        low_similarity[
            [
                "case_id",
                "gold_intent",
                "similarity",
                "relevant",
                "actionable",
                "safe_to_ground"
            ]
        ].to_string(index=False)
    )


# ============================================================
# HIGH-SIMILARITY BUT IRRELEVANT
# ============================================================

print("\n" + "=" * 70)
print("HIGH-SIMILARITY BUT IRRELEVANT")
print("=" * 70)

bad_high_similarity = df[
    (df["relevant"] == "NO")
    &
    (df["similarity"] >= 0.30)
].copy()

print(
    "\nHigh-similarity irrelevant retrievals:",
    len(bad_high_similarity)
)

if len(bad_high_similarity) > 0:

    print(
        bad_high_similarity[
            [
                "case_id",
                "gold_intent",
                "rank",
                "similarity",
                "historical_customer",
                "historical_reply"
            ]
        ]
        .sort_values(
            "similarity",
            ascending=False
        )
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# SAVE CASE SUMMARY
# ============================================================

OUTPUT = "evidence_benchmark_results.csv"

case_summary.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nSaved case-level results:",
    OUTPUT
)

print("\nDone.")
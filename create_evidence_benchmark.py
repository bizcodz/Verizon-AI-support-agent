import pandas as pd
import numpy as np
import pickle
import re

from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

GOLDEN_FILE = "golden_set.csv"

MODEL_FILE = "verizon_resolution_retriever_v3.pkl"

OUTPUT_FILE = "evidence_benchmark.csv"

SAMPLE_SIZE = 50

TOP_K = 3

RANDOM_SEED = 42


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("EVIDENCE QUALITY BENCHMARK")
print("=" * 70)

print("\nLoading golden set...")

gold = pd.read_csv(
    GOLDEN_FILE
)

print(
    "Golden examples:",
    len(gold)
)

print("\nLoading retrieval model...")

with open(
    MODEL_FILE,
    "rb"
) as f:

    model = pickle.load(f)

retrievers = model["retrievers"]


# ============================================================
# SAMPLE GOLDEN CASES
# ============================================================

if len(gold) < SAMPLE_SIZE:

    raise ValueError(
        "Golden set contains fewer than "
        f"{SAMPLE_SIZE} examples."
    )


sample = gold.sample(
    n=SAMPLE_SIZE,
    random_state=RANDOM_SEED
).reset_index(drop=True)


print(
    "Benchmark sample:",
    len(sample)
)


# ============================================================
# RETRIEVE
# ============================================================

rows = []

for _, gold_row in sample.iterrows():

    case_id = gold_row.get(
        "case_id",
        ""
    )

    customer_text = str(
        gold_row["customer_text"]
    )

    gold_intent = str(
        gold_row["final_intent"]
    )

    # IMPORTANT:
    # For the evidence benchmark we use the GOLD intent.
    #
    # This isolates retrieval quality from intent-classifier
    # errors. We are asking:
    #
    # "If we know the correct intent, can retrieval find
    # useful historical evidence?"

    retriever = retrievers.get(
        gold_intent
    )

    if retriever is None:
        continue

    vectorizer = retriever[
        "vectorizer"
    ]

    matrix = retriever[
        "matrix"
    ]

    cases = retriever[
        "cases"
    ]

    query_vector = vectorizer.transform([
        clean_text(customer_text)
    ])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = np.argsort(
        similarities
    )[::-1][:TOP_K]

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        historical_customer = str(
            cases.iloc[idx]["customer_text"]
        )

        historical_reply = str(
            cases.iloc[idx]["support_reply"]
        )

        response_type = str(
            cases.iloc[idx]["response_type"]
        )

        score = float(
            similarities[idx]
        )

        rows.append({

            "case_id": case_id,

            "gold_intent": gold_intent,

            "customer_text": customer_text,

            "rank": rank,

            "similarity": round(
                score,
                4
            ),

            "historical_customer": (
                historical_customer
            ),

            "historical_reply": (
                historical_reply
            ),

            "response_type": response_type,

            # ------------------------------------------------
            # HUMAN ANNOTATION FIELDS
            # ------------------------------------------------

            "relevant": "",

            "actionable": "",

            "safe_to_ground": "",

            "annotation_notes": ""

        })


# ============================================================
# SAVE
# ============================================================

benchmark = pd.DataFrame(
    rows
)

benchmark.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(
    "\nCreated:",
    OUTPUT_FILE
)

print(
    "Rows:",
    len(benchmark)
)

print("\nExpected:")
print(
    SAMPLE_SIZE,
    "cases ×",
    TOP_K,
    "retrievals =",
    SAMPLE_SIZE * TOP_K
)

print("\nDone.")
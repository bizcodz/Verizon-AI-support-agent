import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIG
# ============================================================

DATASET = "twcs.csv"
BRAND = "VerizonSupport"

MODEL_FILE = "verizon_resolution_retriever.pkl"

TOP_K = 5

# Minimum useful customer-message length
MIN_TEXT_LENGTH = 20

# Minimum similarity required before considering evidence useful
MIN_SIMILARITY = 0.15


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# CHECK WHETHER A MESSAGE IS USEFUL FOR RETRIEVAL
# ============================================================

def is_useful_message(text):

    cleaned = clean_text(text)

    if len(cleaned) < MIN_TEXT_LENGTH:
        return False

    # Need at least a few meaningful tokens
    tokens = cleaned.split()

    if len(tokens) < 4:
        return False

    return True


# ============================================================
# BUILD HISTORICAL CUSTOMER → SUPPORT CASES
# ============================================================

def build_cases():

    print("=" * 70)
    print("BUILDING HISTORICAL RESOLUTION INDEX V2")
    print("=" * 70)

    print("\nLoading TWCS dataset...")

    df = pd.read_csv(
        DATASET,
        usecols=[
            "tweet_id",
            "author_id",
            "in_response_to_tweet_id",
            "text"
        ],
        low_memory=False
    )

    print("Total rows:", len(df))

    # Normalize IDs
    df["tweet_id"] = (
        df["tweet_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    df["author_id"] = df["author_id"].astype(str)

    df["in_response_to_tweet_id"] = (
        df["in_response_to_tweet_id"]
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
    )

    # Lookup table
    tweet_lookup = df.set_index("tweet_id")

    # Verizon replies
    verizon = df[df["author_id"] == BRAND].copy()

    print("VerizonSupport replies:", len(verizon))

    cases = []

    for _, row in verizon.iterrows():

        parent_id = row["in_response_to_tweet_id"]

        if parent_id not in tweet_lookup.index:
            continue

        parent = tweet_lookup.loc[parent_id]

        # Parent must be customer-authored
        if str(parent["author_id"]) == BRAND:
            continue

        customer_text = parent["text"]
        support_reply = row["text"]

        if pd.isna(customer_text) or pd.isna(support_reply):
            continue

        customer_text = str(customer_text).strip()
        support_reply = str(support_reply).strip()

        # Remove low-information messages
        if not is_useful_message(customer_text):
            continue

        if len(support_reply) < 5:
            continue

        cases.append({
            "customer_text": customer_text,
            "support_reply": support_reply
        })

    cases_df = pd.DataFrame(cases)

    print("Usable historical cases:", len(cases_df))

    # Deduplicate
    cases_df = cases_df.drop_duplicates(
        subset=["customer_text"]
    ).reset_index(drop=True)

    print("After deduplication:", len(cases_df))

    return cases_df


# ============================================================
# BUILD TF-IDF INDEX
# ============================================================

def build_retriever(cases_df):

    print("\nBuilding TF-IDF index...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        cases_df["customer_text"].map(clean_text)
    )

    print("Index shape:", X.shape)

    model = {
        "vectorizer": vectorizer,
        "matrix": X,
        "cases": cases_df,
        "min_similarity": MIN_SIMILARITY
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("\nSaved retriever:", MODEL_FILE)

    return model


# ============================================================
# RETRIEVE HISTORICAL RESOLUTIONS
# ============================================================

def retrieve(model, query, top_k=TOP_K):

    vectorizer = model["vectorizer"]
    matrix = model["matrix"]
    cases = model["cases"]

    cleaned_query = clean_text(query)

    if len(cleaned_query) < 3:
        return []

    query_vector = vectorizer.transform([cleaned_query])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for rank, idx in enumerate(top_indices, start=1):

        score = float(similarities[idx])

        results.append({
            "rank": rank,
            "score": score,
            "customer_text": cases.iloc[idx]["customer_text"],
            "support_reply": cases.iloc[idx]["support_reply"],
            "usable_evidence": score >= MIN_SIMILARITY
        })

    return results


# ============================================================
# MAIN
# ============================================================cd

if __name__ == "__main__":

    cases = build_cases()

    model = build_retriever(cases)

    print("\n" + "=" * 70)
    print("RETRIEVER V2 TEST")
    print("=" * 70)

    test_queries = [

        "My internet keeps disconnecting and the speed is very slow.",

        "I was charged too much on my Verizon bill.",

        "My Fios TV channels are not working.",

        "Someone opened a Verizon account in my name.",

        "My voicemail is not working."
    ]

    for query in test_queries:

        print("\n" + "-" * 70)
        print("CUSTOMER:")
        print(query)

        results = retrieve(
            model,
            query,
            top_k=3
        )

        for result in results:

            print("\nRank:", result["rank"])
            print(
                "Similarity:",
                round(result["score"], 4)
            )

            print(
                "Usable evidence:",
                result["usable_evidence"]
            )

            print("\nHistorical customer:")
            print(result["customer_text"])

            print("\nHistorical Verizon response:")
            print(result["support_reply"])

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
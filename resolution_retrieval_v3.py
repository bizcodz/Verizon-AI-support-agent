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

MODEL_FILE = "verizon_resolution_retriever_v3.pkl"

TOP_K = 5

MIN_CUSTOMER_LENGTH = 20
MIN_SIMILARITY = 0.12


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# INTENT CLASSIFIER
# ============================================================

def assign_intent(text):

    t = clean_text(text)

    # I10 — Account / fraud
    if any(x in t for x in [
        "fraud",
        "fraudulent",
        "unauthorized",
        "unauthorised",
        "identity theft",
        "stolen identity",
        "someone opened",
        "didn't open",
        "did not open",
        "not me",
        "wasn't me",
        "wasnt me",
        "account hacked",
        "hack my account"
    ]):
        return "I10"

    # I07 — Billing / payment
    if any(x in t for x in [
        "bill",
        "billing",
        "charged",
        "charge",
        "payment",
        "paid",
        "paying",
        "overcharge",
        "overcharged",
        "refund",
        "price",
        "pricing",
        "fee",
        "fees",
        "cost"
    ]):
        return "I07"

    # I08 — Installation / technician
    if any(x in t for x in [
        "install",
        "installation",
        "installer",
        "technician",
        "appointment",
        "schedule",
        "scheduled",
        "no show",
        "no-show",
        "service visit"
    ]):
        return "I08"

    # I09 — Voice / voicemail
    if any(x in t for x in [
        "voicemail",
        "voice mail",
        "landline",
        "home phone",
        "dial tone",
        "phone line"
    ]):
        return "I09"

    # I06 — FiOS app / account access
    if any(x in t for x in [
        "fios app",
        "my fios",
        "fios mobile app",
        "login",
        "log in",
        "logged in",
        "password",
        "authentication",
        "authenticate",
        "sign in",
        "access account",
        "can't access",
        "cannot access"
    ]):
        return "I06"

    # I04 — TV / channels
    if any(x in t for x in [
        "tv",
        "television",
        "channel",
        "channels",
        "dvr",
        "on demand",
        "hbo",
        "showtime",
        "univision",
        "sports",
        "cable"
    ]):
        return "I04"

    # I05 — Mobile service
    if any(x in t for x in [
        "mobile",
        "cell phone",
        "cellular",
        "wireless",
        "4g",
        "5g",
        "lte",
        "text messages",
        "sms"
    ]):
        return "I05"

    # I01 — Internet outage
    if any(x in t for x in [
        "internet is down",
        "internet down",
        "no internet",
        "internet outage",
        "outage",
        "service outage",
        "connection is down",
        "connection down",
        "not working",
        "isn't working",
        "isnt working"
    ]):
        return "I01"

    # I02 — Internet performance
    if any(x in t for x in [
        "slow",
        "slower",
        "slow speed",
        "slow speeds",
        "speed",
        "buffer",
        "buffering",
        "lag",
        "latency",
        "packet loss",
        "disconnect",
        "disconnecting",
        "drops",
        "dropped",
        "unstable",
        "upload",
        "download",
        "ping"
    ]):
        return "I02"

    # I03 — Router / equipment
    if any(x in t for x in [
        "router",
        "modem",
        "ont",
        "equipment",
        "gateway",
        "reboot",
        "reset router",
        "reset modem",
        "red light",
        "lan light",
        "wifi router"
    ]):
        return "I03"

    # I11 — Service complaint
    if any(x in t for x in [
        "customer service",
        "terrible service",
        "horrible service",
        "bad service",
        "ridiculous",
        "unacceptable",
        "complaint",
        "manager",
        "agent",
        "representative",
        "support",
        "waiting",
        "cancel",
        "cancellation"
    ]):
        return "I11"

    return "I12"


# ============================================================
# HISTORICAL RESPONSE QUALITY
# ============================================================

def classify_response(reply):

    text = clean_text(reply)

    # Escalation / channel routing
    if any(x in text for x in [
        "follow and dm",
        "follow us",
        "send us a dm",
        "dm us",
        "direct message",
        "please dm"
    ]):
        return "escalation"

    # Clarifying questions
    if "?" in str(reply):
        return "clarification"

    # Troubleshooting language
    if any(x in text for x in [
        "restart",
        "reset",
        "check",
        "try",
        "troubleshoot",
        "reboot",
        "error",
        "issue",
        "working",
        "connection"
    ]):
        return "troubleshooting"

    # Account action
    if any(x in text for x in [
        "account",
        "billing",
        "charge",
        "payment",
        "refund",
        "activate",
        "cancel"
    ]):
        return "account_action"

    # Very short responses
    if len(text.split()) < 5:
        return "generic"

    return "unknown"


# ============================================================
# BUILD HISTORICAL CASES
# ============================================================

def build_cases():

    print("=" * 70)
    print("BUILDING RETRIEVAL V3")
    print("=" * 70)

    print("\nLoading dataset...")

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

    lookup = df.set_index("tweet_id")

    verizon = df[
        df["author_id"] == BRAND
    ]

    cases = []

    for _, row in verizon.iterrows():

        parent_id = row["in_response_to_tweet_id"]

        if parent_id not in lookup.index:
            continue

        parent = lookup.loc[parent_id]

        if str(parent["author_id"]) == BRAND:
            continue

        customer_text = parent["text"]
        support_reply = row["text"]

        if pd.isna(customer_text) or pd.isna(support_reply):
            continue

        customer_text = str(customer_text).strip()
        support_reply = str(support_reply).strip()

        cleaned_customer = clean_text(customer_text)

        if len(cleaned_customer) < MIN_CUSTOMER_LENGTH:
            continue

        if len(cleaned_customer.split()) < 4:
            continue

        if len(clean_text(support_reply)) < 5:
            continue

        intent = assign_intent(customer_text)

        response_type = classify_response(
            support_reply
        )

        cases.append({
            "customer_text": customer_text,
            "support_reply": support_reply,
            "intent": intent,
            "response_type": response_type
        })

    cases = pd.DataFrame(cases)

    print("\nUsable cases:", len(cases))

    cases = cases.drop_duplicates(
        subset=["customer_text"]
    ).reset_index(drop=True)

    print("After deduplication:", len(cases))

    print("\nIntent distribution:")
    print(
        cases["intent"]
        .value_counts()
        .sort_index()
    )

    print("\nResponse type distribution:")
    print(
        cases["response_type"]
        .value_counts()
    )

    return cases


# ============================================================
# BUILD ONE TF-IDF INDEX PER INTENT
# ============================================================

def build_retriever(cases):

    print("\nBuilding intent-specific indexes...")

    retrievers = {}

    for intent in sorted(cases["intent"].unique()):

        subset = cases[
            cases["intent"] == intent
        ].reset_index(drop=True)

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=50000,
            sublinear_tf=True
        )

        matrix = vectorizer.fit_transform(
            subset["customer_text"].map(clean_text)
        )

        retrievers[intent] = {
            "vectorizer": vectorizer,
            "matrix": matrix,
            "cases": subset
        }

        print(
            intent,
            "->",
            len(subset),
            "cases"
        )

    model = {
        "retrievers": retrievers,
        "min_similarity": MIN_SIMILARITY
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    print("\nSaved:", MODEL_FILE)

    return model


# ============================================================
# RETRIEVE
# ============================================================

def retrieve(model, query, top_k=TOP_K):

    intent = assign_intent(query)

    retriever = model["retrievers"].get(intent)

    if retriever is None:
        return {
            "intent": intent,
            "results": []
        }

    vectorizer = retriever["vectorizer"]
    matrix = retriever["matrix"]
    cases = retriever["cases"]

    query_vector = vectorizer.transform([
        clean_text(query)
    ])

    similarities = cosine_similarity(
        query_vector,
        matrix
    ).flatten()

    # Retrieve more candidates before filtering
    candidate_count = min(
        len(similarities),
        top_k * 5
    )

    indices = np.argsort(
        similarities
    )[::-1][:candidate_count]

    results = []

    for idx in indices:

        score = float(similarities[idx])

        row = cases.iloc[idx]

        results.append({
            "score": score,
            "customer_text": row["customer_text"],
            "support_reply": row["support_reply"],
            "response_type": row["response_type"]
        })

    # Prefer substantive responses
    response_priority = {
        "troubleshooting": 0,
        "account_action": 1,
        "clarification": 2,
        "escalation": 3,
        "unknown": 4,
        "generic": 5
    }

    results.sort(
        key=lambda x: (
            response_priority.get(
                x["response_type"],
                5
            ),
            -x["score"]
        )
    )

    results = results[:top_k]

    # Evidence threshold
    usable = [
        r for r in results
        if r["score"] >= MIN_SIMILARITY
    ]

    return {
        "intent": intent,
        "results": usable
    }


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    cases = build_cases()

    model = build_retriever(cases)

    print("\n" + "=" * 70)
    print("RETRIEVAL V3 TEST")
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

        output = retrieve(
            model,
            query,
            top_k=3
        )

        print("\nPredicted intent:")
        print(output["intent"])

        if not output["results"]:

            print("\nNo sufficiently similar historical evidence.")

            continue

        for rank, result in enumerate(
            output["results"],
            start=1
        ):

            print("\nRank:", rank)

            print(
                "Similarity:",
                round(result["score"], 4)
            )

            print(
                "Response type:",
                result["response_type"]
            )

            print("\nHistorical customer:")
            print(result["customer_text"])

            print("\nHistorical Verizon response:")
            print(result["support_reply"])

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
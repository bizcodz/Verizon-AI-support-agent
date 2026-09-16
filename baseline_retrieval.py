import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

DATASET = "twcs.csv"
GOLDEN = "golden_set.csv"
OUTPUT = "baseline_retrieval_predictions.csv"

BRAND = "VerizonSupport"

INTENTS = [
    "I01", "I02", "I03", "I04",
    "I05", "I06", "I07", "I08",
    "I09", "I10", "I11", "I12"
]

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

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# HEURISTIC LABELING FOR HISTORICAL TRAINING DATA
# ============================================================
#
# These labels are WEAK labels.
# They are NOT used as golden truth.
#
# The 200-case golden set remains completely held out.
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
        "account hacked"
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
        "cost",
        "credit card"
    ]):
        return "I07"

    # I08 — Installation / technician
    if any(x in t for x in [
        "install",
        "installation",
        "installer",
        "technician",
        "tech",
        "appointment",
        "schedule",
        "scheduled",
        "no show",
        "no-show",
        "service visit",
        "coming out"
    ]):
        return "I08"

    # I09 — Voice / voicemail / home phone
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

    # I05 — Mobile phone service
    if any(x in t for x in [
        "mobile",
        "cell phone",
        "cellular",
        "wireless",
        "4g",
        "5g",
        "lte",
        "text messages",
        "sms",
        "phone data"
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
        "wait",
        "waiting",
        "cancel",
        "cancellation"
    ]):
        return "I11"

    return "I12"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("BASELINE 2 — TF-IDF + NEAREST NEIGHBOR RETRIEVAL")
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

print("Total TWCS rows:", len(df))

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

# ============================================================
# FIND VERIZON SUPPORT RESPONSES
# ============================================================

print("\nFinding VerizonSupport responses...")

verizon = df[df["author_id"] == BRAND].copy()

print("VerizonSupport tweets:", len(verizon))

# Map tweet ID -> text/author
tweet_lookup = df.set_index("tweet_id")

# ============================================================
# BUILD HISTORICAL CUSTOMER → VERIZON CASES
# ============================================================

records = []

for _, row in verizon.iterrows():

    parent_id = row["in_response_to_tweet_id"]

    if parent_id in tweet_lookup.index:

        parent = tweet_lookup.loc[parent_id]

        # Only customer-authored parent messages
        if str(parent["author_id"]) != BRAND:

            customer_text = parent["text"]

            if pd.isna(customer_text):
                continue

            customer_text = str(customer_text).strip()

            if len(customer_text) < 5:
                continue

            records.append({
                "customer_text": customer_text,
                "historical_reply": row["text"]
            })


historical = pd.DataFrame(records)

print("Historical customer cases:", len(historical))

# Remove duplicates
historical = historical.drop_duplicates(
    subset=["customer_text"]
).reset_index(drop=True)

print("After deduplication:", len(historical))

# ============================================================
# ASSIGN WEAK LABELS
# ============================================================

print("\nAssigning weak historical labels...")

historical["intent"] = historical["customer_text"].apply(
    assign_intent
)

print("\nHistorical label distribution:")
print(historical["intent"].value_counts().sort_index())

# ============================================================
# LOAD GOLDEN SET
# ============================================================

print("\nLoading golden evaluation set...")

gold = pd.read_csv(GOLDEN)

print("Golden examples:", len(gold))

if "customer_text" not in gold.columns:
    raise ValueError(
        "golden_set.csv must contain a customer_text column."
    )

if "final_intent" not in gold.columns:
    raise ValueError(
        "golden_set.csv must contain a final_intent column."
    )

gold["customer_text"] = gold["customer_text"].fillna("").astype(str)
gold["final_intent"] = gold["final_intent"].astype(str)

# ============================================================
# TF-IDF
# ============================================================

print("\nBuilding TF-IDF representation...")

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    historical["customer_text"].map(clean_text)
)

X_test = vectorizer.transform(
    gold["customer_text"].map(clean_text)
)

print("Training matrix:", X_train.shape)
print("Evaluation matrix:", X_test.shape)

# ============================================================
# KNN RETRIEVAL
# ============================================================

print("\nTraining nearest-neighbor classifier...")

model = KNeighborsClassifier(
    n_neighbors=7,
    weights="distance",
    metric="cosine",
    n_jobs=-1
)

model.fit(
    X_train,
    historical["intent"]
)

predictions = model.predict(X_test)

# ============================================================
# EVALUATION
# ============================================================

y_true = gold["final_intent"].values

accuracy = accuracy_score(y_true, predictions)

macro_f1 = f1_score(
    y_true,
    predictions,
    labels=INTENTS,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    y_true,
    predictions,
    labels=INTENTS,
    average="weighted",
    zero_division=0
)

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

print(f"\nAccuracy:     {accuracy:.4f}")
print(f"Macro F1:     {macro_f1:.4f}")
print(f"Weighted F1:  {weighted_f1:.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_true,
        predictions,
        labels=INTENTS,
        zero_division=0
    )
)

print("\nConfusion matrix:")

cm = confusion_matrix(
    y_true,
    predictions,
    labels=INTENTS
)

cm_df = pd.DataFrame(
    cm,
    index=INTENTS,
    columns=INTENTS
)

print(cm_df)

# ============================================================
# SAVE PREDICTIONS
# ============================================================

results = gold.copy()

results["predicted_intent"] = predictions

results["correct"] = (
    results["final_intent"]
    == results["predicted_intent"]
)

results.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved:", OUTPUT)

print("\nErrors:", (~results["correct"]).sum())
print(
    "Correct:",
    results["correct"].sum(),
    "/",
    len(results)
)

print("\nDone.")
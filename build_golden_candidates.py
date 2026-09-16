import pandas as pd
import re
import random

INPUT_FILE = "twcs.csv"
OUTPUT_FILE = "golden_candidates.csv"

BRAND = "VerizonSupport"
RANDOM_STATE = 42

TARGETS = {
    "I01": 18,
    "I02": 18,
    "I03": 18,
    "I04": 18,
    "I05": 15,
    "I06": 15,
    "I07": 18,
    "I08": 15,
    "I09": 12,
    "I10": 12,
    "I11": 18,
    "I12": 23,
}


def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_id(value):
    """
    Convert tweet IDs into a consistent string representation.

    Handles values such as:
        123456789
        123456789.0
        '123456789'
        '123456789.0'
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Handle Excel/pandas float representation.
    if text.endswith(".0"):
        text = text[:-2]

    return text


def classify_candidate(text):

    t = text.lower()

    scores = {intent: 0 for intent in TARGETS}

    # I01 - Internet outage
    terms = [
        "outage",
        "internet down",
        "internet is down",
        "no internet",
        "internet went down",
        "service down",
        "connection down",
        "network down",
        "out in my area",
        "down in my area",
        "internet outage",
    ]
    scores["I01"] += sum(x in t for x in terms) * 4

    # I02 - Internet performance
    terms = [
        "slow internet",
        "internet slow",
        "slow wifi",
        "slow wi-fi",
        "buffering",
        "lagging",
        "lag ",
        "poor connection",
        "bad connection",
        "unstable",
        "keeps freezing",
        "keeps dropping",
        "speed",
        "slow connection",
    ]
    scores["I02"] += sum(x in t for x in terms) * 3

    # I03 - Router / equipment
    terms = [
        "router",
        "modem",
        "gateway",
        "router reset",
        "reset router",
        "reboot router",
        "rebooted router",
        "cable box",
        "set top box",
        "set-top box",
        "equipment",
        "fios box",
        "box reset",
    ]
    scores["I03"] += sum(x in t for x in terms) * 3

    # I04 - TV / channels
    terms = [
        "tv",
        "television",
        "channel",
        "channels",
        "hbo",
        "comedy central",
        "cable",
        "picture quality",
        "tv picture",
        "programming",
        "show",
        "watch a show",
        "cable box",
    ]
    scores["I04"] += sum(x in t for x in terms) * 3

    # I05 - Mobile phone
    terms = [
        "cell phone",
        "cellphone",
        "mobile phone",
        "mobile",
        "phone",
        "calls",
        "call",
        "text",
        "sms",
        "data",
        "4g",
        "5g",
        "cellular",
        "wireless",
        "replacement phone",
    ]
    scores["I05"] += sum(x in t for x in terms) * 3

    # I06 - FiOS app / access
    terms = [
        "app",
        "application",
        "login",
        "log in",
        "sign in",
        "signin",
        "password",
        "authentication",
        "authenticate",
        "credentials",
        "error 7613",
        "fios app",
        "mobile app",
    ]
    scores["I06"] += sum(x in t for x in terms) * 4

    # I07 - Billing / payment
    terms = [
        "bill",
        "billing",
        "charge",
        "charged",
        "double charge",
        "payment",
        "paid",
        "invoice",
        "credit",
        "refund",
        "fee",
        "price",
        "cost",
        "amount due",
    ]
    scores["I07"] += sum(x in t for x in terms) * 4

    # I08 - Installation / scheduling
    terms = [
        "installation",
        "install",
        "installer",
        "technician",
        "tech",
        "appointment",
        "scheduled",
        "schedule",
        "reschedule",
        "installation date",
        "install date",
        "coming out",
        "service appointment",
    ]
    scores["I08"] += sum(x in t for x in terms) * 4

    # I09 - Voice / voicemail
    terms = [
        "voicemail",
        "voice mail",
        "home phone",
        "landline",
        "land line",
        "dial tone",
        "telephone service",
        "home telephone",
    ]
    scores["I09"] += sum(x in t for x in terms) * 5

    # I10 - Account / fraud
    terms = [
        "fraud",
        "fraudulent",
        "unauthorized",
        "unauthorised",
        "identity theft",
        "stolen identity",
        "someone opened",
        "not my account",
        "account hacked",
        "hacked account",
        "didn't open",
        "did not open",
    ]
    scores["I10"] += sum(x in t for x in terms) * 5

    # I11 - Service complaint
    terms = [
        "worst customer service",
        "terrible customer service",
        "horrible customer service",
        "poor customer service",
        "bad customer service",
        "customer service sucks",
        "unacceptable",
        "ridiculous",
        "disappointed",
        "complaint",
        "complaining",
        "canceling",
        "cancel",
        "fed up",
        "frustrated",
        "frustrating",
        "broken promise",
        "promised",
        "on hold",
    ]
    scores["I11"] += sum(x in t for x in terms) * 2

    max_score = max(scores.values())

    if max_score == 0:
        return "I12", 0

    priority = [
        "I10",
        "I09",
        "I08",
        "I07",
        "I06",
        "I05",
        "I04",
        "I03",
        "I02",
        "I01",
        "I11",
    ]

    best_intent = priority[0]

    for intent in priority:
        if scores[intent] > scores[best_intent]:
            best_intent = intent

    return best_intent, max_score


print("=" * 60)
print("BUILDING VERIZONSUPPORT GOLDEN CANDIDATES")
print("=" * 60)

print("\nReading dataset...")

usecols = [
    "tweet_id",
    "author_id",
    "created_at",
    "in_response_to_tweet_id",
    "text",
]

df = pd.read_csv(
    INPUT_FILE,
    usecols=usecols,
    low_memory=False,
)

df["text"] = df["text"].apply(clean_text)

# ---------------------------------------------------------
# NORMALIZE ALL IDS FIRST
# ---------------------------------------------------------

print("\nNormalizing tweet IDs...")

df["tweet_id_norm"] = df["tweet_id"].apply(normalize_id)

df["response_id_norm"] = df[
    "in_response_to_tweet_id"
].apply(normalize_id)

print(f"Total tweets loaded: {len(df):,}")

# ---------------------------------------------------------
# PASS 1
# ---------------------------------------------------------

print("\nPASS 1: Finding VerizonSupport responses...")

support = df[
    df["author_id"].astype(str).str.strip() == BRAND
].copy()

print(f"VerizonSupport tweets: {len(support):,}")

support = support[
    support["response_id_norm"] != ""
].copy()

parent_ids = set(
    support["response_id_norm"]
)

print(
    f"Unique customer parent IDs: "
    f"{len(parent_ids):,}"
)

# ---------------------------------------------------------
# PASS 2
# ---------------------------------------------------------

print("\nPASS 2: Finding customer parent tweets...")

customers = df[
    df["tweet_id_norm"].isin(parent_ids)
].copy()

customers = customers[
    customers["author_id"].astype(str).str.strip() != BRAND
].copy()

print(
    f"Customer messages found: "
    f"{len(customers):,}"
)

# ---------------------------------------------------------
# BUILD CASES
# ---------------------------------------------------------

print("\nBuilding customer -> VerizonSupport cases...")

support_lookup = (
    support
    .sort_values("created_at")
    .drop_duplicates(
        subset=["response_id_norm"],
        keep="first"
    )
)

cases = customers.merge(
    support_lookup[
        [
            "response_id_norm",
            "tweet_id_norm",
            "tweet_id",
            "created_at",
            "text",
        ]
    ],
    left_on="tweet_id_norm",
    right_on="response_id_norm",
    how="inner",
    suffixes=("_customer", "_support"),
)

cases = cases.rename(
    columns={
        "tweet_id_customer": "customer_tweet_id",
        "created_at_customer": "customer_created_at",
        "text_customer": "customer_text",
        "tweet_id_support": "support_tweet_id",
        "created_at_support": "support_created_at",
        "text_support": "historical_verizon_reply",
    }
)

cases["customer_text"] = cases[
    "customer_text"
].apply(clean_text)

cases["historical_verizon_reply"] = cases[
    "historical_verizon_reply"
].apply(clean_text)

# Remove very short messages.
cases = cases[
    cases["customer_text"].str.len() >= 12
].copy()

# Remove duplicate customer cases.
cases = cases.drop_duplicates(
    subset=["customer_tweet_id"]
).copy()

print(
    f"Candidate cases: "
    f"{len(cases):,}"
)

if len(cases) == 0:
    print("\nERROR: Still found 0 cases.")
    print("Tweet ID matching needs further inspection.")
    raise SystemExit(1)

# ---------------------------------------------------------
# PRE-LABEL ALL CASES
# ---------------------------------------------------------

print("\nPre-labeling ALL candidate cases...")

labels = []
scores = []

for text in cases["customer_text"]:
    label, score = classify_candidate(text)
    labels.append(label)
    scores.append(score)

cases["candidate_intent"] = labels
cases["candidate_score"] = scores

print("\nAVAILABLE CANDIDATES BY INTENT:")

available = cases["candidate_intent"].value_counts()

for intent in TARGETS:
    print(
        f"{intent}: "
        f"{available.get(intent, 0)} available"
    )

# ---------------------------------------------------------
# STRATIFIED SAMPLE
# ---------------------------------------------------------

print("\nSampling golden candidates...")

selected_parts = []

for intent, target in TARGETS.items():

    subset = cases[
        cases["candidate_intent"] == intent
    ].copy()

    if len(subset) <= target:
        chosen = subset
    else:
        chosen = subset.sample(
            n=target,
            random_state=RANDOM_STATE
        )

    selected_parts.append(chosen)

    print(
        f"{intent}: selected {len(chosen)} "
        f"(target {target}, available {len(subset)})"
    )

golden = pd.concat(
    selected_parts,
    ignore_index=True
)

# Shuffle.
golden = golden.sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

# Case IDs.
golden["case_id"] = [
    f"VZ-{i:03d}"
    for i in range(1, len(golden) + 1)
]

# Preserve candidate labels separately.
golden["prelabel_intent"] = golden[
    "candidate_intent"
]

golden["prelabel_confidence"] = golden[
    "candidate_score"
]

# Human-review fields.
golden["intent_label"] = ""
golden["intent_confidence"] = ""
golden["should_escalate"] = ""
golden["escalation_reason"] = ""
golden["expected_resolution"] = ""
golden["review_notes"] = ""

# ---------------------------------------------------------
# FINAL COLUMNS
# ---------------------------------------------------------

final_columns = [
    "case_id",
    "customer_tweet_id",
    "customer_created_at",
    "customer_text",
    "support_tweet_id",
    "support_created_at",
    "historical_verizon_reply",

    "prelabel_intent",
    "prelabel_confidence",

    "intent_label",
    "intent_confidence",
    "should_escalate",
    "escalation_reason",
    "expected_resolution",
    "review_notes",
]

golden = golden[final_columns]

# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

golden.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print(f"Created: {OUTPUT_FILE}")
print(f"Rows: {len(golden)}")

print("\nFINAL PRE-LABEL DISTRIBUTION:")

print(
    golden["prelabel_intent"]
    .value_counts()
    .sort_index()
)

print("\nIMPORTANT:")
print("prelabel_intent is ONLY candidate construction.")
print("It is NOT ground truth.")
print("Final intent_label must be human reviewed.")
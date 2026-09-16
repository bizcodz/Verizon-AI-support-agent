import pandas as pd
import re

INPUT = "golden_candidates.csv"
OUTPUT = "golden_prelabelled.csv"

df = pd.read_csv(INPUT)

print("Pre-labeling candidate cases...")


def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    return text


def classify(text):

    t = clean(text)

    # I10 — Account fraud
    if any(x in t for x in [
        "fraud",
        "fraudulent",
        "identity theft",
        "stolen identity",
        "unauthorized account",
        "unauthorised account",
        "account opened in my name",
        "opened an account in my name",
    ]):
        return "I10"

    # I07 — Billing / payment
    if any(x in t for x in [
        "double charge",
        "charged",
        "charge",
        "bill",
        "billing",
        "payment",
        "amount due",
        "refund",
        "credit",
        "fee",
    ]):
        return "I07"

    # I08 — Installation / scheduling
    if any(x in t for x in [
        "installation",
        "install",
        "technician",
        "appointment",
        "scheduled",
        "schedule",
        "installer",
        "arrival time",
    ]):
        return "I08"

    # I06 — FiOS / Verizon app access
    if any(x in t for x in [
        "fios app",
        "verizon app",
        "mobile app",
        "login",
        "log in",
        "sign in",
        "password",
        "credentials",
        "authentication",
        "error 7613",
    ]):
        return "I06"

    # I09 — Home phone / voicemail
    if any(x in t for x in [
        "voicemail",
        "voice mail",
        "landline",
        "home phone",
        "home telephone",
        "dial tone",
        "phone line",
    ]):
        return "I09"

    # I04 — TV
    if any(x in t for x in [
        "fios tv",
        " tv",
        "television",
        "channel",
        "channels",
        "hbo",
        "hbo go",
        "espn",
        "showtime",
        "on demand",
        "picture quality",
        "tv picture",
        "blackout",
        "watch a show",
        "watching a show",
        "cable",
    ]):
        return "I04"

    # I05 — Mobile phone
    if any(x in t for x in [
        "cell phone",
        "cellphone",
        "mobile phone",
        "wireless phone",
        "mobile data",
        "cell service",
        "cellular",
        "phone service",
        "make a call",
        "can't call",
        "cannot call",
        "phone call",
        "replacement phone",
        "new phone",
    ]):
        return "I05"

    # I03 — Router / equipment
    if any(x in t for x in [
        "router",
        "modem",
        "fios box",
        "set top box",
        "equipment",
        "reboot the router",
        "reset the router",
        "reset router",
        "replace the router",
    ]):
        return "I03"

    # I01 — Internet outage
    if any(x in t for x in [
        "internet went down",
        "internet is down",
        "no internet",
        "internet outage",
        "outage",
        "service is down",
        "internet unavailable",
        "no connection",
        "connection is down",
    ]):
        return "I01"

    # I02 — Internet performance
    if any(x in t for x in [
        "slow internet",
        "internet slow",
        "slow wifi",
        "slow wi-fi",
        "buffering",
        "buffer",
        "unstable",
        "freezing",
        "slow speed",
        "poor speed",
        "speed",
        "mbps",
        "lag",
        "lagging",
        "disconnecting",
        "drops",
        "dropping",
    ]):
        return "I02"

    # I11 — Service complaint
    if any(x in t for x in [
        "worst customer service",
        "terrible customer service",
        "poor customer service",
        "bad customer service",
        "horrible customer service",
        "unacceptable customer service",
        "customer service",
        "on hold",
        "hours on hold",
        "verizon sucks",
        "broken promise",
        "broken promises",
        "lied",
        "lie",
        "frustrated",
        "unacceptable",
        "taking verizon to court",
        "canceling",
        "cancelling",
    ]):
        return "I11"

    return "I12"


# ---------------------------------------------------------
# Pre-label all available candidate cases
# ---------------------------------------------------------

df["prelabel_intent"] = df["customer_text"].apply(classify)


# ---------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------

TARGET_TOTAL = 200

targets = {
    "I01": 17,
    "I02": 17,
    "I03": 17,
    "I04": 17,
    "I05": 17,
    "I06": 17,
    "I07": 17,
    "I08": 17,
    "I09": 12,
    "I10": 12,
    "I11": 17,
    "I12": 30,
}

selected = []

for intent, target in targets.items():

    subset = df[df["prelabel_intent"] == intent]

    n = min(target, len(subset))

    if n > 0:
        sample = subset.sample(
            n=n,
            random_state=42
        )

        selected.append(sample)

        print(
            f"{intent}: selected {n} "
            f"of {len(subset)} available"
        )


golden = pd.concat(
    selected,
    ignore_index=True
)


# ---------------------------------------------------------
# Fill remaining slots if necessary
# ---------------------------------------------------------

if len(golden) < TARGET_TOTAL:

    remaining = df[
        ~df["customer_tweet_id"].isin(
            golden["customer_tweet_id"]
        )
    ]

    extra_needed = TARGET_TOTAL - len(golden)

    extra = remaining.sample(
        n=min(extra_needed, len(remaining)),
        random_state=123
    )

    golden = pd.concat(
        [golden, extra],
        ignore_index=True
    )


# Exactly 200
golden = golden.sample(
    n=min(TARGET_TOTAL, len(golden)),
    random_state=999
).reset_index(drop=True)


# ---------------------------------------------------------
# Case IDs
# IMPORTANT: golden_candidates already has case_id.
# We overwrite it rather than inserting a duplicate.
# ---------------------------------------------------------

golden["case_id"] = [
    f"VZ-{i:03d}"
    for i in range(1, len(golden) + 1)
]


# Move case_id to first column
columns = ["case_id"] + [
    c for c in golden.columns
    if c != "case_id"
]

golden = golden[columns]


# ---------------------------------------------------------
# Annotation columns
# ---------------------------------------------------------

golden["intent_label"] = ""
golden["intent_confidence"] = ""
golden["should_escalate"] = ""
golden["escalation_reason"] = ""
golden["expected_resolution"] = ""
golden["review_notes"] = ""


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

golden.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print("Created:", OUTPUT)
print("Rows:", len(golden))

print()
print("PRE-LABEL DISTRIBUTION:")

print(
    golden["prelabel_intent"]
    .value_counts()
    .sort_index()
)

print()
print("These are PRE-LABELS, not ground truth.")
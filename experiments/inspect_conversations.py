import pandas as pd
from collections import defaultdict
import random

FILE = "twcs.csv"

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "AmericanAir",
    "VirginTrains",
    "GWRHelp",
    "VerizonSupport",
]

# ---------------------------------------------------------
# PASS 1
# Load only the columns we need and collect relevant tweets.
# ---------------------------------------------------------

print("Loading relevant tweets...")

chunks = []

for chunk in pd.read_csv(
    FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "created_at",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id",
    ],
    dtype={
        "tweet_id": "string",
        "author_id": "string",
        "inbound": "boolean",
        "text": "string",
        "response_tweet_id": "string",
        "in_response_to_tweet_id": "string",
    },
    chunksize=100_000,
    low_memory=False,
):
    # Keep tweets made by candidate brands
    brand_rows = chunk[chunk["author_id"].isin(BRANDS)]

    if len(brand_rows):
        chunks.append(brand_rows)

    # Also keep customer tweets that are directly referenced by
    # candidate brand replies.
    parent_ids = set()

    for x in brand_rows["in_response_to_tweet_id"].dropna():
        parent_ids.add(x)

    if parent_ids:
        parent_rows = chunk[chunk["tweet_id"].isin(parent_ids)]
        if len(parent_rows):
            chunks.append(parent_rows)

    print(".", end="", flush=True)

print("\nFinished loading.")

data = pd.concat(chunks, ignore_index=True)

# Remove duplicate rows
data = data.drop_duplicates("tweet_id")

print(f"Relevant tweets loaded: {len(data):,}")

# ---------------------------------------------------------
# Build lookup by tweet ID
# ---------------------------------------------------------

tweets = {}

for _, row in data.iterrows():
    tweets[str(row["tweet_id"])] = {
        "tweet_id": str(row["tweet_id"]),
        "author": str(row["author_id"]),
        "inbound": bool(row["inbound"]),
        "text": str(row["text"]),
        "created_at": str(row["created_at"]),
        "parent": (
            str(row["in_response_to_tweet_id"])
            if pd.notna(row["in_response_to_tweet_id"])
            else None
        ),
        "responses": (
            str(row["response_tweet_id"])
            if pd.notna(row["response_tweet_id"])
            else None
        ),
    }

# ---------------------------------------------------------
# Find conversations
# ---------------------------------------------------------

brand_conversations = defaultdict(list)

for tweet in tweets.values():

    if tweet["author"] not in BRANDS:
        continue

    # Walk backwards through parents
    chain = [tweet]
    current = tweet

    visited = set()

    while current["parent"]:

        parent_id = current["parent"]

        if parent_id in visited:
            break

        visited.add(parent_id)

        if parent_id not in tweets:
            break

        parent = tweets[parent_id]

        chain.append(parent)
        current = parent

    chain.reverse()

    # Only consider conversations containing a customer message
    has_customer = any(
        t["author"] not in BRANDS
        for t in chain
    )

    if has_customer:
        # Root tweet becomes approximate conversation ID
        conversation_id = chain[0]["tweet_id"]

        brand_conversations[
            tweet["author"]
        ].append(chain)

# ---------------------------------------------------------
# Statistics
# ---------------------------------------------------------

print("\n")
print("=" * 100)
print("CONVERSATION ANALYSIS")
print("=" * 100)

for brand in BRANDS:

    conversations = brand_conversations[brand]

    if not conversations:
        print(f"{brand}: no conversations found")
        continue

    lengths = [len(c) for c in conversations]

    multi_turn = sum(
        len(c) >= 4
        for c in conversations
    )

    long_conversations = sum(
        len(c) >= 6
        for c in conversations
    )

    print(f"\n{brand}")
    print("-" * 60)
    print(f"Conversation chains: {len(conversations):,}")
    print(f"Average tweets/chain: {sum(lengths) / len(lengths):.2f}")
    print(f"Median tweets/chain:  {sorted(lengths)[len(lengths)//2]}")
    print(
        f"Multi-turn (4+):     "
        f"{multi_turn:,} "
        f"({multi_turn / len(conversations):.1%})"
    )
    print(
        f"Long (6+):            "
        f"{long_conversations:,} "
        f"({long_conversations / len(conversations):.1%})"
    )

# ---------------------------------------------------------
# Print sample conversations
# ---------------------------------------------------------

print("\n\n")
print("=" * 100)
print("SAMPLE CONVERSATIONS")
print("=" * 100)

random.seed(42)

for brand in BRANDS:

    conversations = brand_conversations[brand]

    # Prefer conversations with 4+ tweets
    good = [
        c for c in conversations
        if len(c) >= 4
    ]

    if not good:
        good = conversations

    if not good:
        continue

    samples = random.sample(
        good,
        min(2, len(good))
    )

    print(f"\n\n######## {brand} ########")

    for number, conversation in enumerate(samples, 1):

        print(f"\n--- Conversation {number} ---")

        for tweet in conversation:

            speaker = (
                "CUSTOMER"
                if tweet["author"] not in BRANDS
                else "BRAND"
            )

            print(
                f"[{speaker}] "
                f"{tweet['text']}"
            )
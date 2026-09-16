import pandas as pd
import re

FILE = "twcs.csv"

TARGET = "VerizonSupport"
OUTPUT = "verizon_customer_sample.csv"

SAMPLE_SIZE = 2000
CHUNK_SIZE = 100_000

samples = []

print("Collecting VerizonSupport customer messages...")

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
    chunksize=CHUNK_SIZE,
    low_memory=False,
):

    # Brand's tweets
    brand = chunk[chunk["author_id"].eq(TARGET)]

    if len(brand) == 0:
        continue

    # Find customer tweets that Verizon directly replied to
    parent_ids = set(
        brand["in_response_to_tweet_id"]
        .dropna()
        .astype(str)
    )

    if not parent_ids:
        continue

    customer_messages = chunk[
        chunk["tweet_id"].isin(parent_ids)
    ].copy()

    # Keep actual customer messages
    customer_messages = customer_messages[
        ~customer_messages["author_id"].eq(TARGET)
    ]

    if len(customer_messages):
        samples.append(customer_messages)

    current = sum(len(x) for x in samples)

    print(f"Collected approximately {current:,} messages...")

    if current >= SAMPLE_SIZE:
        break

if not samples:
    raise RuntimeError("No VerizonSupport customer messages found.")

data = pd.concat(samples, ignore_index=True)

data = data.drop_duplicates("tweet_id")

# Random but reproducible sample
data = data.sample(
    n=min(SAMPLE_SIZE, len(data)),
    random_state=42
)

data = data.sort_values("created_at")

data.to_csv(
    OUTPUT,
    index=False
)

print("\n========================================")
print("DONE")
print("========================================")
print(f"Customer messages: {len(data):,}")
print(f"Saved to: {OUTPUT}")

print("\nExample messages:\n")

for i, text in enumerate(data["text"].head(20), 1):
    print(f"{i}. {text}")
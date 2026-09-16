import pandas as pd
from collections import Counter

FILE = "twcs.csv"
CHUNK_SIZE = 100_000

total_rows = 0
inbound = 0
outbound = 0

columns = None
author_counts = Counter()

print("Profiling TWCS dataset...\n")

for i, chunk in enumerate(
    pd.read_csv(FILE, chunksize=CHUNK_SIZE, low_memory=False)
):
    if columns is None:
        columns = list(chunk.columns)

        print("Columns:")
        for c in columns:
            print(" -", c)
        print()

    total_rows += len(chunk)

    if "inbound" in chunk.columns:
        inbound += (
            chunk["inbound"]
            .astype(str)
            .str.lower()
            .eq("true")
            .sum()
        )

        outbound += (
            chunk["inbound"]
            .astype(str)
            .str.lower()
            .eq("false")
            .sum()
        )

    if "author_id" in chunk.columns:
        author_counts.update(
            chunk["author_id"].astype(str)
        )

    print(f"Processed {(i + 1) * CHUNK_SIZE:,} rows...")

print("\n========== RESULTS ==========")

print(f"Total rows: {total_rows:,}")

if "inbound" in (columns or []):
    print(f"Inbound tweets:  {inbound:,}")
    print(f"Outbound tweets: {outbound:,}")

if author_counts:
    print("\nTop 30 authors:")

    for author, count in author_counts.most_common(30):
        print(f"{author}: {count:,}")
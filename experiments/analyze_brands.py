import pandas as pd
from collections import defaultdict

FILE = "twcs.csv"
CHUNK_SIZE = 100_000

# Brands discovered from the first profiling pass.
BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "SpotifyCares",
    "Delta",
    "Tesco",
    "AmericanAir",
    "TMobileHelp",
    "comcastcares",
    "British_Airways",
    "SouthwestAir",
    "VirginTrains",
    "Ask_Spectrum",
    "XboxSupport",
    "sprintcare",
    "hulu_support",
    "sainsburys",
    "GWRHelp",
    "AskPlayStation",
    "ChipotleTweets",
    "VerizonSupport",
    "UPSHelp",
    "ATVIAssist",
    "O2",
    "Safaricom_Care",
    "idea_cares",
    "AskTarget",
    "AirAsiaSupport",
    "BofA_Help",
    "SW_Help",
]

# Statistics for each brand
stats = {
    brand: {
        "total": 0,
        "inbound": 0,
        "outbound": 0,
        "with_parent": 0,
        "with_response": 0,
        "unique_customers": set(),
    }
    for brand in BRANDS
}

print("Analyzing brands...\n")

for i, chunk in enumerate(
    pd.read_csv(FILE, chunksize=CHUNK_SIZE, low_memory=False)
):
    # Keep only relevant columns
    chunk["author_id"] = chunk["author_id"].astype(str)
    chunk["inbound"] = chunk["inbound"].astype(str).str.lower()

    for brand in BRANDS:
        mask = chunk["author_id"].eq(brand)

        if not mask.any():
            continue

        data = chunk.loc[mask]

        stats[brand]["total"] += len(data)

        incoming = data["inbound"].eq("true")
        outgoing = data["inbound"].eq("false")

        stats[brand]["inbound"] += incoming.sum()
        stats[brand]["outbound"] += outgoing.sum()

        stats[brand]["with_parent"] += (
            data["in_response_to_tweet_id"]
            .notna()
            .sum()
        )

        stats[brand]["with_response"] += (
            data["response_tweet_id"]
            .notna()
            .sum()
        )

        # For inbound tweets, author_id is the customer.
        # However, brand filtering above means we only see brand tweets.
        # So this isn't a customer count; we'll calculate conversation
        # metrics separately below.

    if (i + 1) % 5 == 0:
        print(f"Processed {(i + 1) * CHUNK_SIZE:,} rows...")

print("\n" + "=" * 90)
print("BRAND COMPARISON")
print("=" * 90)

print(
    f"{'Brand':<22}"
    f"{'Tweets':>12}"
    f"{'Inbound':>12}"
    f"{'Outbound':>12}"
    f"{'Parent':>12}"
    f"{'Responses':>12}"
)

print("-" * 90)

for brand, s in sorted(
    stats.items(),
    key=lambda x: x[1]["total"],
    reverse=True
):
    print(
        f"{brand:<22}"
        f"{s['total']:>12,}"
        f"{s['inbound']:>12,}"
        f"{s['outbound']:>12,}"
        f"{s['with_parent']:>12,}"
        f"{s['with_response']:>12,}"
    )

print("\n")
print("Important:")
print(
    "The next step will examine actual customer→brand conversation "
    "threads, because raw tweet count alone is not enough to select "
    "the best brand."
)
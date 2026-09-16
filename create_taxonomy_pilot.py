import pandas as pd

INPUT = "verizon_clustered_messages.csv"
OUTPUT = "taxonomy_pilot.csv"

df = pd.read_csv(INPUT)

print("Loaded:", len(df), "messages")
print("Columns:", df.columns.tolist())

# Sample 3 messages from each cluster.
# groupby.apply() can behave differently across pandas versions,
# so sample each cluster explicitly.
samples = []

for cluster_id in sorted(df["cluster"].dropna().unique()):
    cluster_df = df[df["cluster"] == cluster_id]

    n = min(3, len(cluster_df))

    sample = cluster_df.sample(
        n=n,
        random_state=42
    )

    samples.append(sample)

pilot = pd.concat(
    samples,
    ignore_index=True
)

# Give every pilot example a stable ID
pilot.insert(
    0,
    "pilot_id",
    range(1, len(pilot) + 1)
)

# Add columns for manual labeling
pilot["intent_label"] = ""
pilot["label_confidence"] = ""
pilot["notes"] = ""

# Keep the important columns
pilot = pilot[
    [
        "pilot_id",
        "cluster",
        "tweet_id",
        "created_at",
        "text",
        "intent_label",
        "label_confidence",
        "notes"
    ]
]

# Save Excel-friendly CSV
pilot.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print()
print("=" * 50)
print("SUCCESS")
print("=" * 50)
print(f"Created: {OUTPUT}")
print(f"Examples: {len(pilot)}")

print()
print("Examples per cluster:")
print(
    pilot["cluster"]
    .value_counts()
    .sort_index()
)

print()
print("Next step:")
print(f"Open {OUTPUT} in Excel and label the examples.")
print()
print("intent_label:")
print("I01 - internet_outage")
print("I02 - internet_performance")
print("I03 - router_equipment")
print("I04 - tv_channel_service")
print("I05 - mobile_phone_service")
print("I06 - fios_app_auth")
print("I07 - billing_payment")
print("I08 - installation_scheduling")
print("I09 - voice_voicemail")
print("I10 - account_fraud")
print("I11 - service_complaint")
print("I12 - other_unclear")
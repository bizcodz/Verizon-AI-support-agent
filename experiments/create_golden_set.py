import pandas as pd
import re

INPUT_FILE = "golden_200.txt"
OUTPUT_FILE = "golden_set.csv"

# Approved final labels
approved = {
    "VZ-005":"I07",
    "VZ-006":"I12",
    "VZ-008":"I12",
    "VZ-010":"I02",
    "VZ-012":"I02",
    "VZ-015":"I12",
    "VZ-016":"I04",
    "VZ-018":"I07",
    "VZ-020":"I02",
    "VZ-030":"I04",
    "VZ-034":"I08",
    "VZ-036":"I04",
    "VZ-040":"I12",
    "VZ-041":"I12",
    "VZ-047":"I04",
    "VZ-049":"I10",
    "VZ-051":"I12",
    "VZ-056":"I11",
    "VZ-068":"I12",
    "VZ-069":"I02",
    "VZ-071":"I01",
    "VZ-072":"I03",
    "VZ-073":"I11",
    "VZ-075":"I07",
    "VZ-079":"I02",
    "VZ-080":"I11",
    "VZ-082":"I12",
    "VZ-084":"I06",
    "VZ-086":"I11",
    "VZ-089":"I12",
    "VZ-090":"I08",
    "VZ-096":"I12",
    "VZ-100":"I08",
    "VZ-105":"I02",
    "VZ-108":"I06",
    "VZ-109":"I12",
    "VZ-110":"I09",
    "VZ-117":"I01",
    "VZ-118":"I02",
    "VZ-119":"I12",
    "VZ-120":"I12",
    "VZ-127":"I02",
    "VZ-129":"I12",
    "VZ-132":"I01",
    "VZ-133":"I11",
    "VZ-136":"I10",
    "VZ-139":"I08",
    "VZ-140":"I04",
    "VZ-145":"I12",
    "VZ-150":"I12",
    "VZ-152":"I11",
    "VZ-153":"I12",
    "VZ-161":"I02",
    "VZ-164":"I07",
    "VZ-168":"I01",
    "VZ-169":"I01",
    "VZ-170":"I06",
    "VZ-172":"I02",
    "VZ-174":"I01",
    "VZ-182":"I12",
    "VZ-183":"I12",
    "VZ-184":"I08",
    "VZ-186":"I12",
    "VZ-187":"I04",
    "VZ-188":"I12",
    "VZ-190":"I09",
    "VZ-192":"I03",
    "VZ-196":"I05",
}

# Cases where we accepted a lower-confidence recommendation
medium_confidence = {
    "VZ-008",
    "VZ-012",
    "VZ-015",
    "VZ-040",
    "VZ-068",
    "VZ-073",
    "VZ-075",
    "VZ-082",
    "VZ-089",
    "VZ-096",
    "VZ-100",
    "VZ-105",
    "VZ-110",
    "VZ-119",
    "VZ-129",
    "VZ-145",
    "VZ-176",
    "VZ-184",
    "VZ-189",
    "VZ-199",
}


def parse_file(filename):
    rows = []

    with open(filename, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.rstrip()

        if not line or line.startswith("case_id"):
            continue

        # Locate case ID
        match_case = re.match(r"\s*(VZ-\d+)\s+", line)

        if not match_case:
            continue

        case_id = match_case.group(1)

        # Find the intent token
        match_intent = re.search(r"\bI(?:0[1-9]|1[0-2])\b", line)

        if not match_intent:
            print("WARNING: Could not parse:", case_id)
            continue

        prelabel = match_intent.group(0)

        customer_text = line[
            match_case.end():match_intent.start()
        ].strip()

        historical_reply = line[
            match_intent.end():
        ].strip()

        rows.append({
            "case_id": case_id,
            "customer_text": customer_text,
            "prelabel_intent": prelabel,
            "historical_verizon_reply": historical_reply
        })

    return pd.DataFrame(rows)


print("=" * 70)
print("Creating finalized Hiver golden set")
print("=" * 70)

df = parse_file(INPUT_FILE)

print(f"\nParsed rows: {len(df)}")

if len(df) != 200:
    raise ValueError(
        f"Expected exactly 200 cases, but parsed {len(df)}."
    )

# Apply approved labels.
df["final_intent"] = df["case_id"].map(
    lambda x: approved.get(x, df.loc[df["case_id"] == x, "prelabel_intent"].iloc[0])
)

df["intent_confidence"] = df["case_id"].apply(
    lambda x: "medium" if x in medium_confidence else "high"
)

df["annotation_status"] = "human_approved_model_assisted"

# Sanity check: no missing labels
if df["final_intent"].isna().any():
    missing = df[df["final_intent"].isna()]["case_id"].tolist()
    raise ValueError(f"Missing final labels: {missing}")

# Save
df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nCreated: {OUTPUT_FILE}")

print("\nFinal dataset size:")
print(len(df))

print("\nFinal intent distribution:")
print(
    df["final_intent"]
    .value_counts()
    .sort_index()
    .to_string()
)

print("\nLabel changes from prelabels:")
changes = df[df["prelabel_intent"] != df["final_intent"]]
print(f"{len(changes)} cases changed.")

print("\n" + "=" * 70)
print("GOLDEN SET READY")
print("=" * 70)
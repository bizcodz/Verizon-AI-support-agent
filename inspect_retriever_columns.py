import pickle

PATH = "verizon_resolution_retriever_v3.pkl"

print("=" * 70)
print("RETRIEVER COLUMN INSPECTION")
print("=" * 70)

with open(PATH, "rb") as f:
    data = pickle.load(f)

retrievers = data["retrievers"]

for intent in ["I01", "I02", "I03", "I04", "I07", "I09", "I10", "I11"]:

    print("\n" + "-" * 70)
    print("INTENT:", intent)

    model = retrievers[intent]

    print("Model keys:")
    print(list(model.keys()))

    cases = model.get("cases")

    if cases is not None:
        print("\nCases type:")
        print(type(cases))

        print("\nCases columns:")
        print(list(cases.columns))

        print("\nFirst row:")
        print(cases.iloc[0].to_dict())

    else:
        print("\nNO 'cases' OBJECT FOUND")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
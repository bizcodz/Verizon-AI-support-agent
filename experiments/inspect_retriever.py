import pickle

PATH = "verizon_resolution_retriever_v3.pkl"

print("=" * 70)
print("RETRIEVER STRUCTURE INSPECTION")
print("=" * 70)

with open(PATH, "rb") as f:
    retriever = pickle.load(f)

print("\nTop-level type:")
print(type(retriever))

print("\nTop-level keys:")
if isinstance(retriever, dict):
    print(list(retriever.keys()))

    for key in list(retriever.keys())[:3]:

        print("\n" + "-" * 70)
        print("KEY:", key)

        value = retriever[key]

        print("VALUE TYPE:")
        print(type(value))

        if isinstance(value, dict):
            print("VALUE KEYS:")
            print(list(value.keys()))

            for subkey, subvalue in value.items():
                print(
                    f"  {subkey}: "
                    f"type={type(subvalue)}"
                )

                if hasattr(subvalue, "shape"):
                    print(
                        f"       shape={subvalue.shape}"
                    )

                if hasattr(subvalue, "columns"):
                    print(
                        f"       columns={list(subvalue.columns)}"
                    )

        else:
            if hasattr(value, "shape"):
                print("SHAPE:", value.shape)

            if hasattr(value, "columns"):
                print("COLUMNS:", list(value.columns))

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
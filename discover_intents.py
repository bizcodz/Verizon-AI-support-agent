import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans


INPUT = "verizon_customer_sample.csv"
OUTPUT = "verizon_clustered_messages.csv"

N_CLUSTERS = 12
TOP_TERMS = 12
EXAMPLES_PER_CLUSTER = 8


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv(INPUT)

df = df.dropna(subset=["text"]).copy()

print(f"Loaded {len(df):,} customer messages.")


# =========================================================
# 2. CLEAN TWITTER TEXT
# =========================================================

def clean_text(text):
    text = str(text)

    # URLs
    text = re.sub(r"https?://\S+", " ", text)

    # @mentions
    text = re.sub(r"@\w+", " ", text)

    # Hashtags
    text = re.sub(r"#(\w+)", r"\1", text)

    # HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Extra whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip().lower()


df["clean_text"] = df["text"].apply(clean_text)

# Remove very short messages
df = df[df["clean_text"].str.len() >= 10].copy()

# Reset index
df = df.reset_index(drop=True)

print(f"Messages after cleaning: {len(df):,}")


# =========================================================
# 3. TF-IDF
# =========================================================

vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.90,
    max_features=10_000,
)

X = vectorizer.fit_transform(df["clean_text"])

print(f"TF-IDF matrix: {X.shape}")


# =========================================================
# 4. K-MEANS
# =========================================================

print(f"\nClustering into {N_CLUSTERS} groups...")

model = KMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    n_init=10,
)

df["cluster"] = model.fit_predict(X)


# =========================================================
# 5. GET REPRESENTATIVE EXAMPLES
# =========================================================
#
# Instead of calculating sparse-matrix distances ourselves,
# use cosine similarity to the cluster centroid.
#

terms = vectorizer.get_feature_names_out()

print("\n")
print("=" * 100)
print("DISCOVERED INTENT CLUSTERS")
print("=" * 100)


for cluster_id in range(N_CLUSTERS):

    cluster_mask = df["cluster"] == cluster_id

    cluster_df = df[cluster_mask].copy()

    # -----------------------------------------------------
    # Top terms
    # -----------------------------------------------------

    center = model.cluster_centers_[cluster_id]

    top_indices = center.argsort()[-TOP_TERMS:][::-1]

    top_terms = [
        terms[i]
        for i in top_indices
    ]

    print("\n")
    print("=" * 100)
    print(
        f"CLUSTER {cluster_id} "
        f"({len(cluster_df):,} messages)"
    )
    print("=" * 100)

    print("\nTop terms:")
    print(", ".join(top_terms))

    # -----------------------------------------------------
    # Representative examples
    # -----------------------------------------------------
    #
    # KMeans has cluster labels. We simply take examples
    # closest to the centroid using Euclidean distance
    # after converting ONLY the relevant rows to dense.
    #
    # With ~2,000 messages × 653 features this is safe.
    # -----------------------------------------------------

    positions = cluster_df.index.to_numpy()

    cluster_matrix = X[positions].toarray()

    distances = (
        ((cluster_matrix - center) ** 2)
        .sum(axis=1)
    )

    nearest = distances.argsort()[
        :EXAMPLES_PER_CLUSTER
    ]

    selected = cluster_df.iloc[nearest]

    print("\nRepresentative examples:")

    for i, (_, row) in enumerate(
        selected.iterrows(),
        1
    ):
        print(f"\n{i}. {row['text']}")


# =========================================================
# 6. SAVE RESULTS
# =========================================================

df[
    [
        "tweet_id",
        "created_at",
        "text",
        "clean_text",
        "cluster",
    ]
].to_csv(
    OUTPUT,
    index=False
)


print("\n")
print("=" * 100)
print("DONE")
print("=" * 100)
print(f"Saved clustered dataset to: {OUTPUT}")
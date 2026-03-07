import os
import glob
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import normalize
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Load cleaned CSVs
def load_cleaned_data(path):
    files = glob.glob(os.path.join(path, "*.csv"))
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)

# Tokenize text
def tokenize(df):
    df["tokens"] = df["full_text"].astype(str).apply(
        lambda x: word_tokenize(x.lower())
    )
    return df

# Cluster word vectors into K bins
def cluster_words(model, K):
    vocab = list(model.wv.index_to_key)
    vectors = np.array([model.wv[w] for w in vocab])

    kmeans = KMeans(n_clusters=K, random_state=42)
    labels = kmeans.fit_predict(vectors)

    return {word: labels[i] for i, word in enumerate(vocab)}

# Create document vectors
def doc_vector(tokens, word_to_bin, K):
    vec = np.zeros(K)
    for w in tokens:
        if w in word_to_bin:
            vec[word_to_bin[w]] += 1
    if len(tokens) > 0:
        vec = vec / len(tokens)
    return vec

def generate_vectors(df, word_to_bin, K):
    return np.array([doc_vector(t, word_to_bin, K) for t in df["tokens"]])

# Subreddit purity
def compute_subreddit_purity(df, labels):
    temp = pd.DataFrame({"cluster": labels, "subreddit": df["subreddit"].fillna("unknown")})
    total = len(temp)
    weighted_majority = 0
    for _, group in temp.groupby("cluster"):
        max_count = group["subreddit"].value_counts().iloc[0]
        weighted_majority += max_count
    return weighted_majority / total

# Compute metrics
def compute_metrics(df, vectors, labels):
    vectors_norm = normalize(vectors)

    sil = silhouette_score(vectors_norm, labels, metric="cosine")
    dbi = davies_bouldin_score(vectors_norm, labels)
    ch  = calinski_harabasz_score(vectors_norm, labels)
    pur = compute_subreddit_purity(df, labels)

    return {
        "silhouette_cosine": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": ch,
        "subreddit_purity": pur
    }

# Save output
def save_output(df, vectors, K, out_path):
    os.makedirs(out_path, exist_ok=True)
    out_df = df.copy()

    for i in range(K):
        out_df[f"bin_{i}"] = vectors[:, i]

    out_file = os.path.join(out_path, f"bow_bins_K{K}.csv")
    out_df.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")

# Main
def main():
    CLEANED_PATH = "data/cleaned_csv"
    MODEL_PATH = "data/embedding_data/word2vec/word2vec.model"
    OUTPUT_PATH = "outputs/part2"

    for K in [3, 10, 30]:
        print(f"\n\nRunning BoW-Bins for K = {K}\n\n")

        df = load_cleaned_data(CLEANED_PATH)
        df = tokenize(df)

        model = Word2Vec.load(MODEL_PATH)

        word_to_bin = cluster_words(model, K)
        vectors = generate_vectors(df, word_to_bin, K)

        # Cluster documents using KMeans for metrics
        doc_labels = KMeans(n_clusters=8, random_state=42).fit_predict(vectors)

        metrics = compute_metrics(df, vectors, doc_labels)
        print(f"Metrics for K={K}: {metrics}")

        save_output(df, vectors, K, OUTPUT_PATH)

    print("\nBoW-Bins embedding + metrics complete.")

if __name__ == "__main__":
    main()
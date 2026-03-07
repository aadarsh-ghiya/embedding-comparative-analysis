import os
import glob
import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

# Load all cleaned CSVs
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

# Train Word2Vec
def train_word2vec(tokens, vector_size=100, min_count=2, window=5, epochs=20):
    model = Word2Vec(
        sentences=tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        epochs=epochs
    )
    return model

# Save model + vectors
def save_outputs(model, out_path):
    os.makedirs(out_path, exist_ok=True)
    model.save(os.path.join(out_path, "word2vec.model"))
    model.wv.save_word2vec_format(os.path.join(out_path, "word_vectors.txt"))
    print(f"Saved Word2Vec model + vectors to: {out_path}")

# Main
def main():
    CLEANED_PATH = "data/cleaned_csv"
    OUTPUT_PATH = "data/embedding_data/word2vec"

    print("Loading cleaned CSV files...")
    df = load_cleaned_data(CLEANED_PATH)

    print("Tokenizing text...")
    df = tokenize(df)

    print("Training Word2Vec model...")
    tokens = df["tokens"].tolist()
    model = train_word2vec(tokens)

    print("Saving outputs...")
    save_outputs(model, OUTPUT_PATH)

    print("\nWord2Vec training complete!")

if __name__ == "__main__":
    main()
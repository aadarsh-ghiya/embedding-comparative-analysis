#!/usr/bin/env python3
"""Part 1 pipeline for Lab 8: Doc2Vec embeddings + cosine clustering."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import normalize


SEED = 42
STOPWORDS = set(ENGLISH_STOP_WORDS)

# Three distinct configurations requested in Part 1.
DOC2VEC_CONFIGS = [
    {"name": "vs50_mc2_ep30", "vector_size": 50, "min_count": 2, "epochs": 30},
    {"name": "vs100_mc3_ep40", "vector_size": 100, "min_count": 3, "epochs": 40},
    {"name": "vs200_mc5_ep50", "vector_size": 200, "min_count": 5, "epochs": 50},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Part 1 Doc2Vec embedding and cosine clustering workflow."
    )
    parser.add_argument(
        "--input-dir",
        default="data/cleaned_csv",
        help="Directory containing cleaned CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/part1",
        help="Directory where Part 1 outputs are written.",
    )
    parser.add_argument(
        "--clusters",
        type=int,
        default=8,
        help="Number of clusters for agglomerative clustering.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Top keywords/representative posts to keep per cluster.",
    )
    return parser.parse_args()


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)


def build_text(row: pd.Series) -> str:
    full_text = str(row.get("full_text", "") or "").strip()
    if full_text:
        return full_text

    parts = [
        str(row.get("title_clean", "") or "").strip(),
        str(row.get("body_clean", "") or "").strip(),
        str(row.get("comments_clean", "") or "").strip(),
    ]
    return " ".join(part for part in parts if part)


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z']+", text.lower())
    return [t for t in tokens if len(t) > 1]


def load_documents(input_dir: str) -> pd.DataFrame:
    paths = sorted(Path(input_dir).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found under {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        df = pd.read_csv(path)
        df["source_file"] = path.name
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    merged["text"] = merged.apply(build_text, axis=1)
    merged["tokens"] = merged["text"].fillna("").map(tokenize)
    merged["token_count"] = merged["tokens"].map(len)
    merged = merged[merged["token_count"] >= 3].copy()
    merged.reset_index(drop=True, inplace=True)
    merged["doc_idx"] = merged.index
    return merged


def cluster_cosine(vectors: np.ndarray, n_clusters: int) -> np.ndarray:
    # sklearn changed "affinity" -> "metric"; support both APIs.
    try:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
    except TypeError:
        from sklearn.cluster import AgglomerativeClustering

        model = AgglomerativeClustering(
            n_clusters=n_clusters, affinity="cosine", linkage="average"
        )
    return model.fit_predict(vectors)


def compute_subreddit_purity(df: pd.DataFrame, labels: np.ndarray) -> float:
    temp = pd.DataFrame({"cluster": labels, "subreddit": df["subreddit"].fillna("unknown")})
    total = len(temp)
    weighted_majority = 0.0
    for _, group in temp.groupby("cluster"):
        max_count = group["subreddit"].value_counts().iloc[0]
        weighted_majority += max_count
    return float(weighted_majority / total) if total else 0.0


def top_keywords_for_cluster(
    token_lists: list[list[str]], top_n: int
) -> list[str]:
    counter: Counter[str] = Counter()
    for tokens in token_lists:
        for tok in tokens:
            if tok not in STOPWORDS and len(tok) >= 3:
                counter[tok] += 1
    return [word for word, _ in counter.most_common(top_n)]


def cluster_summary(
    df: pd.DataFrame,
    vectors_normalized: np.ndarray,
    labels: np.ndarray,
    top_n: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []

    for cluster_id in sorted(np.unique(labels)):
        member_idx = np.where(labels == cluster_id)[0]
        members = df.iloc[member_idx]

        cluster_vecs = vectors_normalized[member_idx]
        centroid = cluster_vecs.mean(axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 0:
            centroid = centroid / centroid_norm

        sims = np.dot(cluster_vecs, centroid)
        top_local_positions = np.argsort(-sims)[:top_n]
        rep_idx = member_idx[top_local_positions]

        representatives = []
        for idx in rep_idx:
            row = df.iloc[idx]
            snippet = str(row["text"])[:220].replace("\n", " ").strip()
            representatives.append(
                {
                    "post_id": str(row.get("post_id", "")),
                    "title": str(row.get("title_clean", "")),
                    "snippet": snippet,
                }
            )

        summaries.append(
            {
                "cluster_id": int(cluster_id),
                "size": int(len(member_idx)),
                "keywords": top_keywords_for_cluster(members["tokens"].tolist(), top_n),
                "representative_messages": representatives,
            }
        )

    return summaries


def run_config(
    df: pd.DataFrame,
    config: dict[str, Any],
    n_clusters: int,
    top_n: int,
    output_dir: Path,
) -> dict[str, Any]:
    tagged_docs = [
        TaggedDocument(words=tokens, tags=[str(doc_idx)])
        for doc_idx, tokens in zip(df["doc_idx"], df["tokens"])
    ]

    workers = max(1, (os.cpu_count() or 2) - 1)
    model = Doc2Vec(
        vector_size=config["vector_size"],
        min_count=config["min_count"],
        epochs=config["epochs"],
        window=8,
        dm=1,
        sample=1e-5,
        negative=5,
        seed=SEED,
        workers=workers,
    )
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)

    vectors = np.vstack([model.dv[str(doc_idx)] for doc_idx in df["doc_idx"]])
    vectors_normalized = normalize(vectors, norm="l2")
    labels = cluster_cosine(vectors_normalized, n_clusters=n_clusters)

    result_df = df.copy()
    result_df["cluster_id"] = labels

    cfg_dir = output_dir / config["name"]
    cfg_dir.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "post_id",
        "subreddit",
        "title_clean",
        "body_clean",
        "comments_clean",
        "text",
        "token_count",
        "source_file",
        "cluster_id",
    ]
    result_df[output_cols].to_csv(cfg_dir / "clustered_messages.csv", index=False)
    model.save(str(cfg_dir / "doc2vec.model"))

    summaries = cluster_summary(result_df, vectors_normalized, labels, top_n=top_n)
    with open(cfg_dir / "cluster_summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    metrics = {
        "config_name": config["name"],
        "vector_size": int(config["vector_size"]),
        "min_count": int(config["min_count"]),
        "epochs": int(config["epochs"]),
        "num_documents": int(len(df)),
        "num_clusters": int(n_clusters),
        "silhouette_cosine": float(
            silhouette_score(vectors_normalized, labels, metric="cosine")
        ),
        "calinski_harabasz": float(calinski_harabasz_score(vectors_normalized, labels)),
        "davies_bouldin": float(davies_bouldin_score(vectors_normalized, labels)),
        "subreddit_purity": float(compute_subreddit_purity(df, labels)),
        "cluster_sizes": {
            str(int(cid)): int(size)
            for cid, size in zip(
                *np.unique(labels, return_counts=True)
            )
        },
    }

    with open(cfg_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def rank_configs(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    metrics_df = pd.DataFrame(metrics_list)

    metrics_df["rank_silhouette"] = metrics_df["silhouette_cosine"].rank(
        ascending=False, method="dense"
    )
    metrics_df["rank_db"] = metrics_df["davies_bouldin"].rank(
        ascending=True, method="dense"
    )
    metrics_df["rank_purity"] = metrics_df["subreddit_purity"].rank(
        ascending=False, method="dense"
    )

    metrics_df["aggregate_rank_score"] = (
        metrics_df["rank_silhouette"] + metrics_df["rank_db"] + metrics_df["rank_purity"]
    )

    # Prefer strongest cosine separation, then tie-break by purity and DB index.
    metrics_df = metrics_df.sort_values(
        by=["silhouette_cosine", "subreddit_purity", "davies_bouldin"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best = metrics_df.iloc[0].to_dict()
    return {
        "best_config": best["config_name"],
        "selection_reason": (
            "Highest cosine silhouette score (with purity and Davies-Bouldin as tie-breakers)."
        ),
        "comparison_table": metrics_df.to_dict(orient="records"),
    }


def main() -> None:
    args = parse_args()
    seed_everything()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_documents(args.input_dir)
    if len(df) < args.clusters:
        raise ValueError(
            f"Only {len(df)} valid documents found, but clusters={args.clusters}."
        )

    all_metrics: list[dict[str, Any]] = []
    for config in DOC2VEC_CONFIGS:
        metrics = run_config(
            df=df,
            config=config,
            n_clusters=args.clusters,
            top_n=args.top_n,
            output_dir=output_dir,
        )
        all_metrics.append(metrics)

    ranking = rank_configs(all_metrics)
    summary = {
        "part": "Part 1 - Doc2Vec Embeddings and Cosine Clustering",
        "num_documents_used": int(len(df)),
        "clusters": int(args.clusters),
        "configs_tested": DOC2VEC_CONFIGS,
        "results": all_metrics,
        "best_configuration": ranking["best_config"],
        "selection_reason": ranking["selection_reason"],
        "comparison_table": ranking["comparison_table"],
    }

    with open(output_dir / "part1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Lightweight machine-readable table.
    pd.DataFrame(all_metrics).to_csv(output_dir / "part1_metrics.csv", index=False)

    print(f"Processed {len(df)} documents.")
    print(f"Best configuration: {ranking['best_config']}")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

# Part 1: Doc2Vec Embeddings + Cosine Clustering

## What was done
- Loaded all cleaned Reddit posts from `data/cleaned_csv/*.csv`.
- Built one combined corpus and filtered out very short documents (`<3` tokens).
- Trained **three Doc2Vec configurations**:
  - `vs50_mc2_ep30` (`vector_size=50`, `min_count=2`, `epochs=30`)
  - `vs100_mc3_ep40` (`vector_size=100`, `min_count=3`, `epochs=40`)
  - `vs200_mc5_ep50` (`vector_size=200`, `min_count=5`, `epochs=50`)
- Clustered each embedding set with **cosine distance** using agglomerative clustering (`linkage=average`, `k=8`).
- Compared embedding quality using:
  - cosine silhouette score (higher is better)
  - Davies-Bouldin index (lower is better)
  - Calinski-Harabasz score (higher is better)
  - subreddit purity proxy (higher is better)

Total usable documents: **4,986**.

## Quantitative comparison

| Config | Silhouette (cosine) | Davies-Bouldin | Calinski-Harabasz | Subreddit Purity |
|---|---:|---:|---:|---:|
| `vs50_mc2_ep30` | 0.6571 | 1.2683 | 442.7845 | 0.2361 |
| `vs100_mc3_ep40` | 0.4826 | 1.0807 | 250.9695 | 0.2379 |
| `vs200_mc5_ep50` | 0.3619 | 1.0153 | 98.0154 | 0.2170 |

## Qualitative notes
- All three settings produce one dominant cluster (expected because all data is from related cooking subreddits).
- `vs50_mc2_ep30` gave the strongest cosine separation and the best Calinski-Harabasz score.
- `vs100_mc3_ep40` and `vs200_mc5_ep50` improved Davies-Bouldin slightly but lost substantial silhouette separation.

## Best configuration (Part 1 conclusion)
Selected: **`vs50_mc2_ep30`**.

Reason: it gave the strongest cosine-based cluster separation among the three tested Doc2Vec configurations while retaining similar subreddit purity.

## Outputs generated
- Full summary: `outputs/part1/part1_summary.json`
- Metrics table: `outputs/part1/part1_metrics.csv`
- Per-configuration outputs:
  - `outputs/part1/<config>/doc2vec.model`
  - `outputs/part1/<config>/clustered_messages.csv`
  - `outputs/part1/<config>/cluster_summary.json`
  - `outputs/part1/<config>/metrics.json`

## Re-run command
```bash
python part1_doc2vec.py
```

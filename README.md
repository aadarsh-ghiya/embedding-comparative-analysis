# Embedding-Comparative-Analysis

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

# Part 2: Word2Vec + Bag‑of‑Word‑Bins Embedding

## **Overview**
Part 2 implements an alternative document‑embedding method based on **Word2Vec** and **Bag‑of‑Word‑Bins (BoW‑Bins)**.  
Instead of learning document vectors directly (as in Doc2Vec), this method:

1. Trains a Word2Vec model on all tokens  
2. Clusters word embeddings into **K semantic bins**  
3. Converts each document into a **K‑dimensional normalized frequency vector**  
4. Evaluates embedding quality using clustering metrics  

This allows a direct comparison with Doc2Vec, especially at **3 dimensions**.

---

## **1. Setup Instructions**

### **Create and activate a virtual environment**
```bash
python -m venv env
```

**Windows:**
```bash
env\Scripts\activate
```

**macOS/Linux:**
```bash
source env/bin/activate
```

### **Install dependencies**
```bash
pip install -r requirements.txt
```

Or manually:
```bash
pip install gensim nltk numpy pandas scikit-learn tqdm
```

### **Download NLTK tokenizers**
```bash
python -c "import nltk; nltk.download('punkt')"
```

---

## **2. Method Summary**

### **2.1 Word2Vec Training**
A Word2Vec model is trained on all cleaned Reddit posts from:

- AskCulinary  
- Baking  
- Cooking  
- FoodScience  
- Recipes

### Word2Vec Model 
Binning and Embedding requires a trained Word2Vec model. This model is loaded directly to cluster word embeddings into K bins.
 you must run:
 ```
python part1_doc2vec.py
```
to derive a model for binning and embedding, and the script automatically trains and saves:
```
data/embedding_data/word2vec/word2vec.model
```
This model provides dense word embeddings used for clustering.

### **2.2 Clustering Word Embeddings**
Word vectors are clustered into **K bins** using K‑Means:

- **K = 3** (required for comparison with Doc2Vec‑3D)  
- **K = 10**  
- **K = 30**

Each bin represents a semantic region of the vocabulary.

### **2.3 Creating Document Vectors**
For each document:

1. Tokenize text  
2. Assign each word to a bin  
3. Count bin frequencies  
4. Normalize by document length  

This produces a **K‑dimensional embedding**.

Example (K=3):
```
[0.6034, 0.3103, 0.0862]
```

---

## **3. Running Binning and Embedding using the Word2Vec model**

Run the script:

```bash
python part2_bow_bins.py
```

This will:

- Load cleaned CSVs  
- Load the trained Word2Vec model  
- Generate BoW‑Bins embeddings for K = 3, 10, 30  
- Compute clustering metrics  
- Save outputs under `outputs/part2/`

---

## **4. Outputs Generated**

For each K, the script produces:

```
outputs/part2/bow_bins_K3.csv
outputs/part2/bow_bins_K10.csv
outputs/part2/bow_bins_K30.csv
```

Each file contains:

- Original document metadata  
- K‑dimensional embedding columns (`bin_0`, `bin_1`, …)  

Metrics are printed to the console:

- Silhouette Score  
- Davies–Bouldin Index  
- Calinski–Harabasz Score  
- Subreddit Purity  

---

## **5. Results Summary**

| K | Silhouette ↑ | DBI ↓ | CH ↑ | Purity ↑ |
|---|--------------|--------|--------|-----------|
| **3** | **0.6263** | **0.5640** | **24651.83** | **0.4509** |
| **10** | 0.3546 | 1.4700 | 1046.15 | 0.5427 |
| **30** | 0.2179 | 2.2031 | 430.70 | 0.6280 |

### **Interpretation**
- **K = 3** gives the strongest structural clustering (best silhouette, DBI, CH).  
- **K = 30** gives the highest purity but weaker separation.  
- Increasing K increases granularity but reduces cluster compactness.

---

## **6. Comparison with Doc2Vec (3 Dimensions)**

To match dimensionality:

- **Doc2Vec → vector_size = 3**  
- **BoW‑Bins → K = 3**

| Method | Dim | Silhouette ↑ | DBI ↓ | CH ↑ | Purity ↑ |
|--------|------|--------------|--------|--------|-----------|
| Doc2Vec (3D) | 3 | 0.3926 | 0.9662 | 1610.46 | 0.2601 |
| BoW‑Bins (K=3) | 3 | **0.6263** | **0.5640** | **24651.83** | **0.4509** |

### **Conclusion**
At **3 dimensions**, BoW‑Bins (K=3) outperforms Doc2Vec‑3D across all metrics.  
However, Doc2Vec becomes superior at higher dimensions (50D, 100D, 200D).

## Part 3 - Comparative Analysis (README)
**Overview**
Part 3 provides a critical comparison between the two embedding methods used in this lab:
- Doc2Vec (Part 1)
- Word2Vec + Bag‑of‑Word‑Bins (BoW‑Bins) (Part 2)
Both methods were evaluated using multiple dimensions/bins to understand how they behave as the representation size increases.

1. Evaluation Methods Used
To compare the two embedding approaches, we used four clustering quality metrics:
- Silhouette Score (cosine) - measures how well clusters are separated
- Davies–Bouldin Index - measures cluster compactness
- Calinski–Harabasz Score - measures between‑cluster vs within‑cluster variance
- Subreddit Purity - measures how well clusters align with subreddit labels
These metrics were chosen because they capture both geometric structure (silhouette, DBI, CH) and semantic grouping (purity), giving a balanced evaluation of embedding quality.

2. Dimension‑Matched Comparison (3D vs K=3)
To ensure a fair comparison, both methods were evaluated at 3 dimensions:
- Doc2Vec --> vector_size = 3
- BoW‑Bins --> K = 3
Result:
BoW‑Bins (K=3) outperformed Doc2Vec‑3D across all metrics.
Doc2Vec becomes too compressed at 3 dimensions, while BoW‑Bins retains enough distributional information to form clearer clusters.

3. Behavior Across Higher Dimensions
- Doc2Vec improves significantly at moderate dimensions (50D), producing the best overall structure and semantic grouping.
- BoW‑Bins becomes more sparse and noisy as K increases (10, 30), improving purity but losing structural separation.

4. Which Method Better Represents Meaning?
- Doc2Vec is better at capturing semantic meaning when allowed higher dimensionality (50D).
- BoW‑Bins is surprisingly strong at very low dimensions (3D) but does not model context or deeper semantics.

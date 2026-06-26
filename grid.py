"""
hdbscan_grid_search.py
=======================
Grid-searches UMAP + HDBSCAN hyperparameters to find the combination that
gives the best cluster separation (silhouette score) for your ticket
embeddings, instead of hand-guessing min_cluster_size / min_samples.

REQUIRES: ticket_embeddings.npy (already produced by cluster.py STEP 5)
          -> run cluster.py at least once first so this file exists.

OUTPUT: hdbscan_grid_search_results.csv  (full grid, every combination)
        + a printed top-20 table, and a filtered top-20 table
          (noise < 5%, clusters > 10) so you don't just chase the
          highest number blindly.

EFFICIENCY NOTE:
UMAP is the expensive step. HDBSCAN re-clustering on an already-reduced
embedding is fast. So instead of re-running UMAP inside the innermost
loop (which the naive version of this grid search would do), we run
UMAP ONCE per n_components value, cache the result, and reuse it for
every min_cluster_size / min_samples / epsilon combination. This turns
~72 UMAP runs into just 3.
"""

import time
from itertools import product

import numpy as np
import pandas as pd
import umap
import hdbscan
from sklearn.metrics import silhouette_score

SEP = "=" * 90

# ======================================================================
# LOAD PRECOMPUTED EMBEDDINGS
# ======================================================================

print("Loading embeddings...")
X = np.load("ticket_embeddings.npy")
print(f"  Embeddings shape: {X.shape}")

# ======================================================================
# GRID DEFINITION  (~72 combinations — well within the 50-100 you wanted)
# ======================================================================

n_components_list = [5, 10, 15]
min_cluster_sizes  = [10, 20, 30, 40]
min_samples_list   = [3, 5, 10]
epsilons           = [0.0, 0.1]

N_NEIGHBORS = 20   # kept fixed, matching your original cluster.py STEP 6

combos = list(product(n_components_list, min_cluster_sizes, min_samples_list, epsilons))
print(f"\nTotal combinations to test: {len(combos)}")

# ======================================================================
# STEP 1 — Precompute UMAP ONCE per n_components value (expensive step)
# ======================================================================

print(f"\n{SEP}\nSTEP 1 — PRECOMPUTING UMAP (one run per n_components)\n{SEP}")

umap_cache = {}
for n_comp in n_components_list:
    t0 = time.time()
    print(f"  Running UMAP  n_components={n_comp} ...", end=" ", flush=True)
    reducer = umap.UMAP(
        n_neighbors=N_NEIGHBORS,
        n_components=n_comp,
        metric="cosine",
        min_dist=0.0,
        random_state=42,
    )
    umap_cache[n_comp] = reducer.fit_transform(X)
    print(f"done in {time.time()-t0:.1f}s  -> shape {umap_cache[n_comp].shape}")

# ======================================================================
# STEP 2 — Grid search over HDBSCAN params (fast, reuses cached UMAP)
# ======================================================================

print(f"\n{SEP}\nSTEP 2 — HDBSCAN GRID SEARCH\n{SEP}")

results = []

for i, (n_comp, mcs, ms, eps) in enumerate(combos, 1):
    X_umap = umap_cache[n_comp]

    try:
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="euclidean",
            cluster_selection_epsilon=eps,
            cluster_selection_method="eom",
            prediction_data=True,
        )
        labels = clusterer.fit_predict(X_umap)

        mask        = labels != -1
        n_clusters  = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise     = int((labels == -1).sum())
        noise_pct   = 100 * n_noise / len(labels)

        if n_clusters > 1 and mask.sum() > n_clusters:
            score = silhouette_score(X_umap[mask], labels[mask])
        else:
            score = -1.0

    except Exception as e:
        n_clusters, noise_pct, score = 0, 100.0, -1.0
        print(f"  [{i}/{len(combos)}] FAILED ({n_comp},{mcs},{ms},{eps}): {e}")

    results.append({
        "n_components"    : n_comp,
        "min_cluster_size": mcs,
        "min_samples"     : ms,
        "epsilon"         : eps,
        "clusters"        : n_clusters,
        "noise_percent"   : round(noise_pct, 2),
        "silhouette"      : round(score, 4),
    })

    if i % 10 == 0 or i == len(combos):
        print(f"  [{i}/{len(combos)}] tested...")

results_df = pd.DataFrame(results)
results_df.to_csv("hdbscan_grid_search_results.csv", index=False)
print(f"\nFull grid saved -> hdbscan_grid_search_results.csv")

# ======================================================================
# STEP 3 — REPORT
# ======================================================================

print(f"\n{SEP}\nTOP 20 BY SILHOUETTE SCORE (unfiltered — sanity-check these, "
      f"don't just trust the number)\n{SEP}")
print(
    results_df.sort_values("silhouette", ascending=False)
    .head(20)
    .to_string(index=False)
)

print(f"\n{SEP}\nTOP 20 — FILTERED  (noise < 5%, clusters > 10)\n{SEP}")
filtered = results_df[
    (results_df["noise_percent"] < 5) &
    (results_df["clusters"] > 10)
].sort_values("silhouette", ascending=False)

if len(filtered):
    print(filtered.head(20).to_string(index=False))
else:
    print("No combination matched (noise < 5% AND clusters > 10).")
    print("Try relaxing the filter, e.g. noise < 10 or clusters > 5.")

# ======================================================================
# STEP 4 — RECOMMENDATION
# ======================================================================

print(f"\n{SEP}\nRECOMMENDATION\n{SEP}")
if len(filtered):
    best = filtered.iloc[0]
    print(
        "Best balanced config found:\n"
        f"  n_components      = {int(best['n_components'])}\n"
        f"  min_cluster_size  = {int(best['min_cluster_size'])}\n"
        f"  min_samples       = {int(best['min_samples'])}\n"
        f"  epsilon           = {best['epsilon']}\n"
        f"  -> clusters={int(best['clusters'])}, "
        f"noise={best['noise_percent']}%, silhouette={best['silhouette']}\n\n"
        "Plug these into cluster.py:\n"
        f"  STEP 6 UMAP   -> n_components={int(best['n_components'])}\n"
        f"  STEP 7 HDBSCAN -> min_cluster_size={int(best['min_cluster_size'])}, "
        f"min_samples={int(best['min_samples'])}, "
        f"cluster_selection_epsilon={best['epsilon']}"
    )
else:
    top = results_df.sort_values("silhouette", ascending=False).iloc[0]
    print(
        "No config satisfied the strict filter, here's the single highest "
        "silhouette score instead (inspect its noise% before trusting it):\n"
        f"  n_components={int(top['n_components'])}, "
        f"min_cluster_size={int(top['min_cluster_size'])}, "
        f"min_samples={int(top['min_samples'])}, epsilon={top['epsilon']} "
        f"-> clusters={int(top['clusters'])}, noise={top['noise_percent']}%, "
        f"silhouette={top['silhouette']}"
    )
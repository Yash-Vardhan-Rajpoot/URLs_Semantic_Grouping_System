# ==========================================================
# clustering.py
# Semantic Clustering Module
# ==========================================================

import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


# ==========================================================
# AUTO THRESHOLD SELECTION
# ==========================================================

def find_best_threshold(embeddings):

    thresholds = np.arange(
        0.35,
        0.65,
        0.05
    )

    best_score = -1
    best_threshold = 0.45

    for threshold in thresholds:

        try:

            clusterer = AgglomerativeClustering(
                n_clusters=None,
                metric="cosine",
                linkage="average",
                distance_threshold=threshold
            )

            labels = clusterer.fit_predict(
                embeddings
            )

            n_clusters = len(
                np.unique(labels)
            )

            # Skip bad clustering results
            if (
                n_clusters <= 1
                or
                n_clusters >= len(labels)
            ):
                continue

            score = silhouette_score(
                embeddings,
                labels,
                metric="cosine"
            )

            if score > best_score:

                best_score = score
                best_threshold = threshold

        except:
            pass

    return best_threshold


# ==========================================================
# MAIN CLUSTERING
# ==========================================================

def perform_clustering(
    embeddings,
    sim_matrix
):

    n_urls = len(embeddings)

    # ------------------------------------------------------
    # Small Dataset Handling
    # ------------------------------------------------------

    if n_urls <= 10:

        threshold = 0.40

    else:

        threshold = find_best_threshold(
            embeddings
        )

    # ------------------------------------------------------
    # Agglomerative Clustering
    # ------------------------------------------------------

    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=threshold
    )

    labels = clusterer.fit_predict(
        embeddings
    )

    # ------------------------------------------------------
    # Singleton Detection
    # ------------------------------------------------------

    next_cluster = (
        np.max(labels) + 1
    )

    for i in range(len(labels)):

        sims = np.delete(
            sim_matrix[i],
            i
        )

        max_similarity = np.max(
            sims
        )

        # If URL is weakly related
        if max_similarity < 0.50:

            labels[i] = next_cluster

            next_cluster += 1

    # ------------------------------------------------------
    # Re-number clusters
    # Example:
    # [5,5,9,9,12]
    # =>
    # [0,0,1,1,2]
    # ------------------------------------------------------

    unique_clusters = sorted(
        np.unique(labels)
    )

    mapping = {
        old: new
        for new, old
        in enumerate(unique_clusters)
    }

    labels = np.array(
        [
            mapping[label]
            for label in labels
        ]
    )

    return labels, threshold
"""Training-zone PCA, cluster selection, and stability diagnostics."""
import numpy as np
import pandas as pd
from scipy.cluster.vq import ClusterError, kmeans2
from scipy.spatial.distance import cdist, pdist, squareform


def fit_kmeans(values, k, seed, restarts):
    """Return the lowest-inertia successful fit over seeded k-means++ starts."""
    generator = np.random.default_rng(seed)
    best = None
    for _ in range(restarts):
        try:
            centers, _ = kmeans2(values, k, iter=100, minit="++", missing="raise", rng=generator)
        except (ClusterError, np.linalg.LinAlgError):
            continue
        distances = cdist(values, centers, "sqeuclidean")
        labels = distances.argmin(axis=1)
        if len(np.unique(labels)) != k:
            continue
        inertia = float(distances.min(axis=1).sum())
        if best is None or inertia < best[2]:
            best = (centers, labels, inertia)
    if best is None:
        raise ValueError(f"No nonempty k-means solution for k={k}.")
    return best


def silhouette(distances, labels):
    """Mean silhouette, assigning zero to points in singleton clusters."""
    groups = [labels == group for group in np.unique(labels)]
    averages = np.stack([distances[:, group].mean(axis=1) for group in groups], axis=1)
    scores = np.zeros(len(labels))
    for column, group in enumerate(groups):
        n = group.sum()
        if n == 1:
            continue
        within = averages[group, column] * n / (n - 1)
        between = np.delete(averages[group], column, axis=1).min(axis=1)
        denominator = np.maximum(within, between)
        scores[group] = np.divide(
            between - within, denominator, out=np.zeros(n), where=denominator > 0,
        )
    return float(scores.mean())


def adjusted_rand(first, second):
    joint = pd.crosstab(first, second).to_numpy(dtype=float)
    def pairs(counts):
        return (counts * (counts - 1) / 2).sum()

    row_pairs, column_pairs = pairs(joint.sum(axis=1)), pairs(joint.sum(axis=0))
    expected = row_pairs * column_pairs / pairs(np.array([len(first)]))
    maximum = (row_pairs + column_pairs) / 2
    return float((pairs(joint) - expected) / (maximum - expected)) if maximum != expected else 1.0


def analyze_profiles(profiles, config):
    values = profiles.to_numpy()
    matrix = values - values.mean(axis=0)
    _, singular, axes = np.linalg.svd(matrix, full_matrices=False)
    signs = np.sign(axes[np.arange(len(axes)), np.abs(axes).argmax(axis=1)])
    axes *= signs[:, None]
    variance = singular ** 2 / (singular ** 2).sum()
    coords = matrix @ axes[:2].T
    pairs = squareform(pdist(values))
    seed = config["seed"]
    restarts = config["kmeans_restarts"]
    evaluations, fitted = [], {}
    for k in config["candidate_k"]:
        centers, labels, inertia = fit_kmeans(values, k, seed, restarts)
        fitted[k] = (centers, labels)
        evaluations.append((k, silhouette(pairs, labels), inertia, np.bincount(labels).min()))
    selection = pd.DataFrame(evaluations, columns=["k", "silhouette", "inertia", "smallest_cluster"])
    k = int(selection.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    centers, labels = fitted[k]
    # Keep label order reproducible by increasing centroid peak hour.
    order = np.argsort(centers.argmax(axis=1), kind="stable")
    labels = np.argsort(order)[labels]
    centers = centers[order]
    scaled = matrix / values.std(axis=0)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(1, config["stability_replicates"] + 1):
        trial_seed = int(rng.integers(0, 2**32))
        _, label_trial, _ = fit_kmeans(values, k, trial_seed, restarts)
        n = round(len(values) * config["zone_subsample_share"])
        subset = np.sort(rng.choice(len(values), n, replace=False))
        sub_centers, _, _ = fit_kmeans(values[subset], k, trial_seed, restarts)
        projected = cdist(values, sub_centers).argmin(axis=1)
        _, scaled_labels, _ = fit_kmeans(scaled, k, trial_seed, restarts)
        rows.extend([
            ("different_seed", replicate, trial_seed, len(values), adjusted_rand(labels, label_trial)),
            ("80_percent_zones", replicate, trial_seed, n, adjusted_rand(labels, projected)),
            ("standardized_hours", replicate, trial_seed, len(values), adjusted_rand(labels, scaled_labels)),
        ])
    stability = pd.DataFrame(rows, columns=["check", "replicate", "seed", "fit_zones", "adjusted_rand"])
    zones = pd.DataFrame({"cluster": labels + 1, "pc1": coords[:, 0], "pc2": coords[:, 1]},
                         index=profiles.index)
    centroids = pd.DataFrame(centers, columns=profiles.columns)
    centroids.index = pd.Index(np.arange(1, k + 1), name="cluster")
    return selection, zones, centroids, stability, variance, axes

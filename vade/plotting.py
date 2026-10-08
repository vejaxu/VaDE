"""Original-space plotting adapted from ../IDEC/idec_reproduction/plotting.py."""
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/opencode/vade-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits

from .datasets import load_original


def sample_indices(labels, limit, seed):
    if len(labels) <= limit:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    unique, counts = np.unique(labels, return_counts=True)
    if limit < len(unique):
        return np.sort(rng.choice(len(labels), limit, replace=False))
    selected = []
    remaining = limit
    for position, (label, count) in enumerate(zip(unique, counts)):
        take = remaining if position == len(unique) - 1 else max(1, int(round(limit * count / len(labels))))
        take = min(take, remaining - (len(unique) - position - 1))
        candidates = np.flatnonzero(labels == label)
        chosen = rng.choice(candidates, min(take, len(candidates)), replace=False)
        selected.append(chosen)
        remaining -= len(chosen)
    indices = np.concatenate(selected)
    if len(indices) < limit:
        unused = np.setdiff1d(np.arange(len(labels)), indices)
        indices = np.concatenate([indices, rng.choice(unused, limit - len(indices), replace=False)])
    return np.sort(indices[:limit])


def project(values, seed):
    if values.shape[1] == 1:
        return np.column_stack([values[:, 0], np.zeros(len(values))])
    if values.shape[1] == 2 or len(values) < 3:
        return values[:, :2]
    return TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto",
                perplexity=min(30., len(values) - 1)).fit_transform(values)


def scatter(values, labels, path):
    unique, encoded = np.unique(labels, return_inverse=True)
    if len(unique) <= 10:
        colors = [plt.get_cmap("tab10")(i) for i in range(len(unique))]
    elif len(unique) <= 20:
        colors = [plt.get_cmap("tab20")(i) for i in range(len(unique))]
    else:
        colors = plt.cm.hsv(np.linspace(0, 1, len(unique)))
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(values[:, 0], values[:, 1], c=encoded, cmap=ListedColormap(colors), alpha=.7, s=15)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dataset(name, data_dir, output_dir, seed=42, sample_size=5000, threads=4):
    if sample_size < 1:
        raise ValueError("plot sample size must be positive")
    output = Path(output_dir)
    raw, truth, _, _, _ = load_original(name, data_dir)
    predictions = np.load(output / "labels_aligned.npy")
    if len(predictions) != len(truth):
        raise ValueError("Saved predictions and original data have different lengths")
    indices = sample_indices(truth, sample_size, seed)
    with threadpool_limits(limits=threads):
        # No MinMax, standardization or model embedding is applied to raw features.
        coordinates = project(np.asarray(raw[indices], dtype=np.float32), seed)
    scatter(coordinates, predictions[indices], output / "original_space_clustering.jpg")
    scatter(coordinates, truth[indices], output / "original_space_true_labels.jpg")
    np.savez(output / "plot_data.npz", sample_indices=indices, coordinates=coordinates,
             predicted_labels=predictions[indices], true_labels=truth[indices])
    return dict(feature_space="original unscaled features", projection="identity" if raw.shape[1] <= 2 else "t-SNE",
                samples=len(indices), seed=seed, sample_limit=sample_size)

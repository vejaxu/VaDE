"""Hungarian-aligned clustering metrics. No labels enter model fitting."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, f1_score, normalized_mutual_info_score


def align_labels(truth, predictions):
    classes, encoded = np.unique(truth, return_inverse=True)
    clusters, cluster_ids = np.unique(predictions, return_inverse=True)
    table = np.zeros((len(clusters), len(classes)), dtype=np.int64)
    np.add.at(table, (cluster_ids, encoded), 1)
    rows, cols = linear_sum_assignment(-table)
    # Unmatched clusters remain distinct, so they cannot spuriously improve F1.
    mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    for r in range(len(clusters)):
        if r not in mapping:
            mapping[r] = len(classes) + r
    aligned = np.array([mapping[int(i)] for i in cluster_ids], dtype=np.int64)
    details = {str(clusters[r].item()): mapping[r] for r in range(len(clusters))}
    return encoded, aligned, details


def evaluate(truth, predictions):
    encoded, aligned, mapping = align_labels(truth, predictions)
    scores = dict(nmi=float(normalized_mutual_info_score(encoded, aligned)),
                  ari=float(adjusted_rand_score(encoded, aligned)),
                  f1_macro=float(f1_score(encoded, aligned, average="macro", zero_division=0)))
    return scores, aligned, mapping

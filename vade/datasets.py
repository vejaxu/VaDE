"""Unified DC dataset loading with feature-wise MinMax scaling."""
import pickle
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

DATASETS = [
    "spiral", "AC", "4C", "RingG", "complex9", "USPS", "STL-10", "Cifar-10",
    "ImageNet-10", "ImageNet-Dogs", "MNIST", "COIL20", "w1Gaussians", "w100Gaussians",
    "sparse_3_dense_3_dense_3", "sparse_8_dense_1_dense_1", "one_gaussian_10_one_line_5_2",
    "tutorial", "tonsil", "airway", "crohn", "dlpfc_151507", "non_spherical",
    "non_spherical_gap", "non_spherical_gap_0_5", "non_spherical_gap_0_8",
]
ALIASES = {"MNIST": "mnist.mat", "dlpfc_151507": "151507_final.pkl",
           **{name: f"processed_{name}.pkl" for name in ("tutorial", "tonsil", "airway", "crohn")}}


def resolve_dataset(name, data_dir):
    root = Path(data_dir)
    candidates = [root / ALIASES.get(name, f"{name}.mat"), root / f"{name}.pkl"]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Dataset {name}: none of {[str(p) for p in candidates]} exists")


def load_original(name, data_dir):
    path = resolve_dataset(name, data_dir)
    if path.suffix == ".mat":
        content = loadmat(path)
        feature_key = "data" if "data" in content else "X"
        label_key = "class" if "class" in content else "Y"
    else:
        with path.open("rb") as f:
            content = pickle.load(f)
        feature_key, label_key = "expression_normalized", "ground_truth"
    if feature_key not in content or label_key not in content:
        raise ValueError(f"{path}: expected features {feature_key} and labels {label_key}")
    raw = content[feature_key]
    if sparse.issparse(raw):
        raw = raw.toarray()
    raw = np.asarray(raw)
    labels = np.asarray(content[label_key]).reshape(-1)
    if raw.ndim != 2 or len(raw) != len(labels) or len(raw) < 2:
        raise ValueError(f"{path}: incompatible features {raw.shape} and labels {labels.shape}")
    if not np.isfinite(raw).all():
        raise ValueError(f"{path}: non-finite features")
    # Preserve all rows and original labels; no label-dependent sample filtering.
    if labels.dtype.kind in "fc" and not np.isfinite(labels).all():
        raise ValueError(f"{path}: non-finite ground-truth labels")
    return raw, labels, path, feature_key, label_key


def load_dataset(name, data_dir):
    raw, labels, path, feature_key, label_key = load_original(name, data_dir)
    classes, encoded = np.unique(labels, return_inverse=True)
    if len(classes) < 2:
        raise ValueError(f"{path}: fewer than two ground-truth classes")
    scaler = MinMaxScaler()
    features = scaler.fit_transform(raw.astype(np.float32)).clip(0, 1)
    metadata = dict(dataset=name, source=str(path.resolve()), feature_key=feature_key,
                    label_key=label_key, samples=len(raw), input_dim=raw.shape[1],
                    n_clusters=len(classes), normalization="feature-wise MinMax [0, 1]",
                    class_values=classes.tolist())
    return (np.ascontiguousarray(features, dtype=np.float32), labels, encoded.astype(np.int64),
            scaler, metadata)

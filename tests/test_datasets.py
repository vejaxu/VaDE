import numpy as np
from scipy.io import savemat

from vade.datasets import load_dataset
from vade.metrics import evaluate


def test_minmax_keeps_order_and_constant_columns(tmp_path):
    raw = np.array([[-2., 7.], [0., 7.], [2., 7.]])
    savemat(tmp_path / "test.mat", dict(data=raw, **{"class": np.array([[2, 1, 2]])}))
    x, labels, encoded, scaler, meta = load_dataset("test", tmp_path)
    np.testing.assert_allclose(x, [[0, 0], [.5, 0], [1, 0]])
    np.testing.assert_array_equal(labels, [2, 1, 2])
    assert meta["n_clusters"] == 2


def test_hungarian_alignment_before_macro_f1():
    scores, aligned, mapping = evaluate(np.array([10, 10, 20, 20]), np.array([1, 1, 0, 0]))
    assert scores == dict(nmi=1., ari=1., f1_macro=1.)
    np.testing.assert_array_equal(aligned, [0, 0, 1, 1])

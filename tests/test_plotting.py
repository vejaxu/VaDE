import numpy as np
from scipy.io import savemat

from vade.plotting import plot_dataset, sample_indices


def test_plot_uses_original_coordinates_and_saved_alignment(tmp_path):
    raw = np.array([[-20., 300.], [-10., 500.], [100., -40.], [200., -80.]])
    truth = np.array([1, 1, 2, 2])
    savemat(tmp_path / "test.mat", {"data": raw, "class": truth})
    np.save(tmp_path / "labels_aligned.npy", [0, 0, 1, 1])
    metadata = plot_dataset("test", tmp_path, tmp_path, sample_size=10)
    saved = np.load(tmp_path / "plot_data.npz")
    np.testing.assert_array_equal(saved["coordinates"], raw)
    assert metadata["projection"] == "identity"
    assert (tmp_path / "original_space_clustering.jpg").exists()
    assert (tmp_path / "original_space_true_labels.jpg").exists()


def test_plot_sampling_is_reproducible_and_preserves_classes():
    labels = np.repeat([0, 1, 2], [100, 30, 5])
    indices = sample_indices(labels, 30, 42)
    np.testing.assert_array_equal(indices, sample_indices(labels, 30, 42))
    assert len(np.unique(indices)) == 30
    assert len(np.unique(labels[indices])) == 3

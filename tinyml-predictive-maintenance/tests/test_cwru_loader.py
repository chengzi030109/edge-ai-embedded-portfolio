"""Tests for the CWRU loader.

A real CWRU download is huge and not always reachable, so we fabricate a
CWRU-shape ``.mat`` file in a temp dir. The loader should pick up the
``X{nnn}_DE_time`` channel and slice windows the same way it does for real
files.
"""

import numpy as np
import pytest

scipy_io = pytest.importorskip("scipy.io")

from tpm.datasets.cwru import load_all, load_label


def _write_mat(path, key, signal):
    scipy_io.savemat(str(path), {key: signal.reshape(-1, 1)}, do_compression=True)


def test_load_label_extracts_windows(tmp_path):
    """A single .mat with a DE channel becomes non-overlapping windows."""

    folder = tmp_path / "normal"
    folder.mkdir()
    rng = np.random.default_rng(0)
    signal = rng.normal(size=4096).astype(np.float64)
    _write_mat(folder / "X097_DE_time.mat", "X097_DE_time", signal)

    ws = load_label(tmp_path, "normal", window_size=1024)

    assert ws.label == "normal"
    assert ws.windows.shape == (4, 1024)
    assert ws.source_files == ("X097_DE_time.mat",)


def test_load_all_handles_multiple_labels(tmp_path):
    """The convenience helper loads each requested label folder."""

    rng = np.random.default_rng(1)
    for label, key in [("normal", "X097_DE_time"), ("inner", "X105_DE_time")]:
        folder = tmp_path / label
        folder.mkdir()
        _write_mat(folder / f"{key}.mat", key, rng.normal(size=2048).astype(np.float64))

    sets = load_all(tmp_path, labels=("normal", "inner"), window_size=1024)
    assert set(sets.keys()) == {"normal", "inner"}
    assert sets["normal"].windows.shape[1] == 1024
    assert sets["inner"].windows.shape[1] == 1024


def test_missing_label_folder_raises(tmp_path):
    """A missing folder is a clear configuration error, not silent zeros."""

    with pytest.raises(FileNotFoundError):
        load_label(tmp_path, "ball", window_size=1024)

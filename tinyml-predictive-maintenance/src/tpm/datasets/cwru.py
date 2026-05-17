"""CWRU bearing dataset loader.

The Case Western Reserve University bearing dataset is the de-facto reference
for condition-monitoring research. It contains accelerometer recordings from a
test rig with healthy bearings and seeded faults at several locations, fault
diameters, and motor loads.

Why this matters here:
- Evaluating a detector only on its own synthetic data is not credible.
- CWRU windows look noticeably different from the synthetic motor signal in
  ``signal_sim``, so passing this evaluation is a real test of the feature
  extraction and model code, not just of the simulator.

This loader is filesystem-first: it accepts any ``.mat`` files dropped under
``data/cwru/`` and never assumes the CWRU servers are reachable. A separate
prep script can populate that directory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

# CWRU drive-end accelerometer sample rate. The 12 kHz set is the most widely
# cited; the 48 kHz set exists for some files but we standardize on 12 kHz to
# keep the comparison consistent.
CWRU_SAMPLE_RATE_HZ = 12_000

# CWRU file naming convention varies, but every .mat file we use exposes one
# accelerometer channel whose key ends with one of these suffixes.
_CHANNEL_SUFFIXES = ("_DE_time", "_FE_time", "_BA_time")


@dataclass(frozen=True)
class CwruWindowSet:
    """Container for windowed CWRU recordings of a single label.

    Fields:
        label: ``"normal"`` or one of ``"inner"``, ``"outer"``, ``"ball"``.
        windows: 2-D float32 array shaped ``(n_windows, window_size)``.
        source_files: file names the windows were extracted from.
    """

    label: str
    windows: np.ndarray
    source_files: tuple[str, ...]


def _load_mat_channel(path: Path) -> np.ndarray:
    """Return the first accelerometer time-series found inside a ``.mat`` file.

    CWRU files store the relevant signal under keys like ``X097_DE_time``. The
    numeric prefix changes per file, so we match on the suffix. Drive-end (DE)
    is preferred because most CWRU papers report results on it.
    """

    # ``scipy.io`` is heavy. Import lazily so users who never touch CWRU do not
    # pay the import cost.
    from scipy.io import loadmat

    payload = loadmat(str(path), squeeze_me=True)
    for suffix in _CHANNEL_SUFFIXES:
        for key, value in payload.items():
            if key.endswith(suffix) and hasattr(value, "shape"):
                return np.asarray(value, dtype=np.float32).reshape(-1)
    raise ValueError(
        f"no DE/FE/BA accelerometer channel found in {path.name}; "
        "this file may be from a different dataset"
    )


def _slice_windows(signal: np.ndarray, window_size: int, hop: int) -> np.ndarray:
    """Cut a 1-D signal into fixed-size windows with the given hop.

    A non-overlapping schedule (``hop == window_size``) keeps windows
    independent, which matters when reporting evaluation metrics. The caller
    can request overlap to grow the training set.
    """

    if signal.size < window_size:
        return np.empty((0, window_size), dtype=np.float32)
    n_windows = 1 + (signal.size - window_size) // hop
    out = np.empty((n_windows, window_size), dtype=np.float32)
    for i in range(n_windows):
        start = i * hop
        out[i] = signal[start : start + window_size]
    return out


def load_label(
    root: str | Path,
    label: str,
    window_size: int = 1024,
    hop: int | None = None,
) -> CwruWindowSet:
    """Load all .mat files for one label under ``root/<label>/`` into windows.

    Layout expected on disk::

        data/cwru/
          normal/   *.mat
          inner/    *.mat
          outer/    *.mat
          ball/     *.mat

    The user (or ``scripts/prepare_cwru.py``) is responsible for placing files
    into the right folder. Misclassified files would corrupt the evaluation,
    so we surface a clear error if a label folder is missing or empty.
    """

    folder = Path(root) / label
    if not folder.is_dir():
        raise FileNotFoundError(
            f"missing CWRU label folder: {folder}. "
            "Run scripts/prepare_cwru.py or drop .mat files manually."
        )

    files = sorted(folder.glob("*.mat"))
    if not files:
        raise FileNotFoundError(f"no .mat files in {folder}")

    hop = hop if hop is not None else window_size

    chunks: list[np.ndarray] = []
    sources: list[str] = []
    for path in files:
        signal = _load_mat_channel(path)
        windows = _slice_windows(signal, window_size, hop)
        if windows.size:
            chunks.append(windows)
            sources.append(path.name)

    if not chunks:
        raise ValueError(
            f"no usable windows extracted for label '{label}'. "
            "Files may be shorter than window_size."
        )

    return CwruWindowSet(
        label=label,
        windows=np.concatenate(chunks, axis=0),
        source_files=tuple(sources),
    )


def load_all(
    root: str | Path,
    labels: Iterable[str] = ("normal", "inner", "outer", "ball"),
    window_size: int = 1024,
    hop: int | None = None,
) -> dict[str, CwruWindowSet]:
    """Load every requested label. Helper for end-to-end scripts."""

    return {
        label: load_label(root, label, window_size=window_size, hop=hop)
        for label in labels
    }

from __future__ import annotations

"""Dataset adapters for public industrial audio folders.

Real anomaly datasets rarely share one perfect layout. MIMII-style data often
looks like ``fan/id_00/normal/*.wav`` and ``fan/id_00/abnormal/*.wav``; smaller
examples often use ``train/normal`` and ``test/anomaly``. This adapter keeps the
rules simple and inspectable: recursively scan WAV files, infer labels from path
segments, infer split names when present, and leave unknown files out.
"""

from pathlib import Path

from .features import extract_features, read_wav

NORMAL_NAMES = {"normal", "healthy", "ok"}
ANOMALY_NAMES = {"abnormal", "anomaly", "anomalous", "fault", "faulty", "broken"}
TRAIN_NAMES = {"train", "training"}
TEST_NAMES = {"test", "testing", "eval", "evaluation", "dev"}


def load_public_audio_rows(root: str | Path) -> list[dict]:
    """Load labeled WAV rows from a recursive public-dataset-style folder.

    Returned rows use the same dictionary style as the original demo loader, but
    include extra metadata useful for reports: ``split``, ``relative_path``, and
    ``machine_id``. A file is labeled by the closest matching path segment. For
    example, ``fan/id_00/abnormal/0001.wav`` becomes ``label="anomaly"``.
    """

    base = Path(root)
    rows: list[dict] = []
    for path in sorted(base.rglob("*.wav")):
        relative = path.relative_to(base)
        parts = [part.lower() for part in relative.parts[:-1]]
        label = _infer_label(parts)
        if label is None:
            continue
        samples, sample_rate_hz = read_wav(path)
        rows.append(
            {
                "path": str(path),
                "relative_path": str(relative),
                "label": label,
                "split": _infer_split(parts),
                "machine_id": _infer_machine_id(parts),
                "features": extract_features(samples, sample_rate_hz),
                "sample_rate_hz": sample_rate_hz,
            }
        )
    return rows


def split_train_eval_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Choose one-class training rows and evaluation rows.

    Public anomaly datasets normally train only on healthy audio. If an explicit
    train/test split exists, train on ``train`` normal rows and evaluate on
    ``test`` rows. If the folder has only normal/abnormal directories, train on
    the first half of normal files and evaluate on the remaining normal files
    plus every anomaly file. This fallback keeps small local fixtures useful.
    """

    train_normals = [row for row in rows if row["split"] == "train" and row["label"] == "normal"]
    eval_rows = [row for row in rows if row["split"] == "test"]
    if train_normals and eval_rows:
        return train_normals, eval_rows

    normals = [row for row in rows if row["label"] == "normal"]
    anomalies = [row for row in rows if row["label"] == "anomaly"]
    cutoff = max(1, len(normals) // 2)
    return normals[:cutoff], normals[cutoff:] + anomalies


def summarize_rows(rows: list[dict]) -> dict:
    """Return small dataset counts for Markdown and JSON reports."""

    labels = {"normal": 0, "anomaly": 0}
    splits: dict[str, int] = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    return {"count": len(rows), "labels": labels, "splits": splits}


def _infer_label(parts: list[str]) -> str | None:
    for part in reversed(parts):
        if part in NORMAL_NAMES:
            return "normal"
        if part in ANOMALY_NAMES:
            return "anomaly"
    return None


def _infer_split(parts: list[str]) -> str:
    for part in parts:
        if part in TRAIN_NAMES:
            return "train"
        if part in TEST_NAMES:
            return "test"
    return "all"


def _infer_machine_id(parts: list[str]) -> str:
    for part in parts:
        if part.startswith("id_"):
            return part
    return "default"

from __future__ import annotations

"""Streaming-style audio analysis.

The demo still reads WAV files from disk, but this module processes them as
sliding windows. That mirrors an embedded Linux service consuming microphone,
ALSA, or UDP audio chunks and emitting anomaly events over time.
"""

import time
from pathlib import Path
from typing import Iterable

import numpy as np

from .alarm import AlarmDebouncer
from .backends import AudioModelBackend
from .features import extract_features, iter_windows, read_wav
from .synth import write_wav


def collect_window_feature_vectors(
    rows: Iterable[dict],
    window_seconds: float,
    hop_seconds: float,
    *,
    label_filter: str | None = "normal",
) -> list[np.ndarray]:
    """Collect sliding-window feature vectors for model training.

    The first version of the demo trained the centroid model on one feature
    vector per full WAV clip, then evaluated on short streaming windows. That
    works as a smoke test, but it creates a train/inference mismatch: a 1 second
    clip has smoother statistics than a 0.25 second online window. Embedded
    audio services normally train and infer on the same window shape, so this
    helper builds the training matrix from the exact same windowing contract
    used by ``analyze_wav_windows``.

    ``label_filter="normal"`` keeps the usual one-class anomaly-detection
    setup: learn healthy machine sound only, then flag windows far from that
    centroid. Passing ``None`` is useful in tests or future supervised
    baselines where every row should contribute.
    """

    vectors: list[np.ndarray] = []
    for row in rows:
        if label_filter is not None and row.get("label") != label_filter:
            continue

        samples, sample_rate_hz = read_wav(row["path"])
        for window in iter_windows(samples, sample_rate_hz, window_seconds, hop_seconds):
            vectors.append(extract_features(window["samples"], sample_rate_hz))
    return vectors


def analyze_wav_windows(
    path: str | Path,
    label: str,
    model: AudioModelBackend,
    window_seconds: float,
    hop_seconds: float,
    *,
    clips_dir: str | Path | None = None,
    save_anomaly_clips: bool = False,
    debouncer: AlarmDebouncer | None = None,
) -> list[dict]:
    """Analyze one WAV file as a stream of overlapping windows.

    ``debouncer`` is optional so unit tests and one-off analysis can inspect raw
    model output without state. The dataset replay passes a shared debouncer
    across files to mimic a long-running process where alarm state survives from
    one input chunk to the next.
    """

    samples, sample_rate_hz = read_wav(path)
    events: list[dict] = []
    alarm_filter = debouncer or AlarmDebouncer(on_count=1, off_count=1)
    clip_dir = Path(clips_dir) if clips_dir is not None else None
    if clip_dir is not None and save_anomaly_clips:
        clip_dir.mkdir(parents=True, exist_ok=True)

    for index, window in enumerate(iter_windows(samples, sample_rate_hz, window_seconds, hop_seconds)):
        start = time.perf_counter()
        features = extract_features(window["samples"], sample_rate_hz)
        feature_ms = (time.perf_counter() - start) * 1000.0
        pred_start = time.perf_counter()
        pred = model.predict(features)
        inference_ms = (time.perf_counter() - pred_start) * 1000.0
        alarm = alarm_filter.update(bool(pred["is_anomaly"]))
        clip_path = ""
        if save_anomaly_clips and pred["is_anomaly"] and clip_dir is not None:
            clip_path = str(clip_dir / f"{Path(path).stem}_win{index:03d}.wav")
            write_wav(clip_path, window["samples"], sample_rate_hz)
        events.append(
            {
                "source": str(path),
                "label": label,
                "window_index": index,
                "start_s": float(window["start_s"]),
                "end_s": float(window["end_s"]),
                "score": pred["score"],
                "threshold": pred["threshold"],
                "is_anomaly_raw": pred["is_anomaly"],
                "is_anomaly": pred["is_anomaly"],
                "is_alarm": alarm["is_alarm"],
                "alarm_state": alarm["alarm_state"],
                "alarm_bad_streak": alarm["alarm_bad_streak"],
                "alarm_good_streak": alarm["alarm_good_streak"],
                "feature_ms": feature_ms,
                "inference_ms": inference_ms,
                "clip_path": clip_path,
                "features": {name: float(value) for name, value in zip(model.feature_names, features, strict=False)},
            }
        )
    return events


def analyze_dataset_windows(
    rows: Iterable[dict],
    model: AudioModelBackend,
    window_seconds: float,
    hop_seconds: float,
    *,
    clips_dir: str | Path | None = None,
    save_anomaly_clips: bool = False,
    alarm_on_count: int = 3,
    alarm_off_count: int = 5,
) -> list[dict]:
    """Analyze every WAV file row and flatten all window events.

    The debouncer is created once for the whole replay. That is intentional: a
    deployed service does not reset alarm state just because the next PCM chunk
    came from a different file in our demo folder.
    """

    events: list[dict] = []
    debouncer = AlarmDebouncer(on_count=alarm_on_count, off_count=alarm_off_count)
    for row in rows:
        events.extend(
            analyze_wav_windows(
                row["path"],
                row["label"],
                model,
                window_seconds,
                hop_seconds,
                clips_dir=clips_dir,
                save_anomaly_clips=save_anomaly_clips,
                debouncer=debouncer,
            )
        )
    return events

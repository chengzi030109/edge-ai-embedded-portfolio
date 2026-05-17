"""Numerical parity test between Python and C centroid inference.

The firmware/inference.c port is a direct mirror of model.py. This test
compiles the C harness, feeds it random feature vectors, and confirms that
the score and is_anomaly flag match the Python detector to within floating
point rounding.

The test is skipped when:
  - no C compiler is on PATH (typical local Windows dev),
  - no model.json exists in artifacts/.

CI runs on Linux and Windows have gcc/cl.exe available, so this is exercised
in the cloud even when local development cannot build C.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tpm.model import CentroidAnomalyDetector

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware"
ARTIFACTS = ROOT / "artifacts"
MODEL_JSON = ARTIFACTS / "model.json"


def _find_compiler() -> tuple[str, list[str]] | None:
    """Return (compiler, extra args) or None if no compiler is available."""

    for name in ("gcc", "cc", "clang"):
        path = shutil.which(name)
        if path:
            return path, ["-O2", "-std=c99", "-lm"]
    cl = shutil.which("cl")
    if cl:
        # MSVC: different flag syntax. Math is in the standard library.
        return cl, ["/O2", "/nologo"]
    return None


@pytest.fixture(scope="module")
def compiled_harness(tmp_path_factory) -> Path:
    """Compile firmware/test_inference.c and return the executable path."""

    compiler = _find_compiler()
    if compiler is None:
        pytest.skip("no C compiler on PATH (gcc/cc/clang/cl)")
    if not MODEL_JSON.exists():
        pytest.skip(f"missing {MODEL_JSON}; run scripts/train_model.py first")

    # Regenerate the model_params.h header so the C side reflects the current
    # JSON model. This keeps the test honest if someone retrained without
    # re-exporting.
    subprocess.run(
        [
            "python",
            str(ROOT / "scripts" / "export_model_to_c.py"),
            "--model",
            str(MODEL_JSON),
            "--out",
            str(FIRMWARE / "model_params.h"),
        ],
        check=True,
        cwd=ROOT,
    )

    out_dir = tmp_path_factory.mktemp("c_build")
    exe = out_dir / ("test_inference.exe" if compiler[0].endswith(".exe") else "test_inference")

    cc, flags = compiler
    if "cl" in Path(cc).name.lower():
        # MSVC: combine sources, write executable.
        cmd = [
            cc,
            *flags,
            "-I",
            str(FIRMWARE),
            str(FIRMWARE / "test_inference.c"),
            str(FIRMWARE / "inference.c"),
            f"/Fe:{exe}",
        ]
    else:
        cmd = [
            cc,
            *flags,
            "-I",
            str(FIRMWARE),
            str(FIRMWARE / "test_inference.c"),
            str(FIRMWARE / "inference.c"),
            "-o",
            str(exe),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"C compilation failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return exe


def _run_c(exe: Path, vector: np.ndarray) -> tuple[float, bool]:
    """Pipe a feature vector to the C harness and parse its output."""

    payload = struct.pack(f"<{len(vector)}f", *vector.astype(np.float32))
    proc = subprocess.run([str(exe)], input=payload, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"C harness exited with {proc.returncode}: {proc.stderr.decode(errors='replace')}"
        )
    score_str, flag_str = proc.stdout.decode().strip().split()
    return float(score_str), bool(int(flag_str))


def test_c_inference_matches_python(compiled_harness):
    """Random feature vectors should produce matching score + is_anomaly."""

    detector = CentroidAnomalyDetector.load(MODEL_JSON)
    rng = np.random.default_rng(0)

    # Mix near-mean (likely normal) and far-from-mean (likely anomaly) vectors
    # so both sides of the threshold are exercised.
    samples = []
    samples.append(detector.mean.copy())
    for _ in range(8):
        samples.append(detector.mean + rng.normal(0.0, 0.05, size=detector.mean.shape))
    for _ in range(8):
        samples.append(detector.mean + rng.normal(0.0, detector.scale * 5.0, size=detector.mean.shape))

    for vector in samples:
        py_result = detector.predict(vector)
        c_score, c_flag = _run_c(compiled_harness, vector)
        assert c_score == pytest.approx(py_result["score"], rel=1e-5, abs=1e-5)
        assert c_flag == py_result["is_anomaly"]

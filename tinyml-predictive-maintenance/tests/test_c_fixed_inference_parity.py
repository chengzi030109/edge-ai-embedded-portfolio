"""Parity test for the Q24.8 fixed-point C inference path.

The float C path proves the algorithm was ported correctly. This test goes one
step closer to MCU deployment: Python quantizes the model and input features,
the C harness runs integer-only inference, and both sides must return the same
Q24.8 score and anomaly decision.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tpm.fixed_point import FixedPointCentroid
from tpm.model import CentroidAnomalyDetector

ROOT = Path(__file__).resolve().parents[1]
FIRMWARE = ROOT / "firmware"
ARTIFACTS = ROOT / "artifacts"
MODEL_JSON = ARTIFACTS / "model.json"


def _find_compiler() -> tuple[str, list[str]] | None:
    """Return a C compiler command or ``None`` when local C tests must skip."""

    for name in ("gcc", "cc", "clang"):
        path = shutil.which(name)
        if path:
            # TinyCC is a compact Windows fallback installed as cc.exe in this
            # workspace. It accepts the project sources directly without GCC
            # optimization/standard flags.
            if (Path(path).parent / "tcc.exe").exists():
                return path, []
            return path, ["-O2", "-std=c99"]
    cl = shutil.which("cl")
    if cl:
        return cl, ["/O2", "/nologo"]
    return None


@pytest.fixture(scope="module")
def compiled_fixed_harness(tmp_path_factory) -> Path:
    """Compile firmware/test_inference_fixed.c and return the executable path."""

    compiler = _find_compiler()
    if compiler is None:
        pytest.skip("no C compiler on PATH (gcc/cc/clang/cl)")
    if not MODEL_JSON.exists():
        pytest.skip(f"missing {MODEL_JSON}; run scripts/train_model.py first")

    # Generate both headers so this test can be run after any retraining step.
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "export_model_to_c.py"),
            "--model",
            str(MODEL_JSON),
            "--out",
            str(FIRMWARE / "model_params.h"),
            "--fixed-out",
            str(FIRMWARE / "model_params_fixed.h"),
        ],
        check=True,
        cwd=ROOT,
    )

    out_dir = tmp_path_factory.mktemp("c_fixed_build")
    exe = out_dir / ("test_inference_fixed.exe" if compiler[0].endswith(".exe") else "test_inference_fixed")

    cc, flags = compiler
    if "cl" in Path(cc).name.lower():
        cmd = [
            cc,
            *flags,
            "-I",
            str(FIRMWARE),
            str(FIRMWARE / "test_inference_fixed.c"),
            str(FIRMWARE / "inference_fixed.c"),
            f"/Fe:{exe}",
        ]
    else:
        cmd = [
            cc,
            *flags,
            "-I",
            str(FIRMWARE),
            str(FIRMWARE / "test_inference_fixed.c"),
            str(FIRMWARE / "inference_fixed.c"),
            "-o",
            str(exe),
        ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        pytest.fail(f"fixed-point C compilation failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    return exe


def _run_c_fixed(exe: Path, vector: np.ndarray) -> tuple[int, bool]:
    """Pipe float features to the fixed C harness and parse Q-score output."""

    payload = struct.pack(f"<{len(vector)}f", *vector.astype(np.float32))
    proc = subprocess.run([str(exe)], input=payload, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"fixed C harness exited with {proc.returncode}: {proc.stderr.decode(errors='replace')}"
        )
    score_q_str, flag_str = proc.stdout.decode().strip().split()
    return int(score_q_str), bool(int(flag_str))


def test_fixed_c_inference_matches_python_integer_path(compiled_fixed_harness):
    """C Q24.8 score and decision should match Python's integer reference."""

    detector = CentroidAnomalyDetector.load(MODEL_JSON)
    fixed = FixedPointCentroid.from_float_model(detector)
    rng = np.random.default_rng(42)

    samples = [detector.mean.copy()]
    for _ in range(8):
        samples.append(detector.mean + rng.normal(0.0, 0.1, size=detector.mean.shape))
    for _ in range(8):
        samples.append(detector.mean + rng.normal(0.0, detector.scale * 6.0, size=detector.mean.shape))

    for vector in samples:
        py = fixed.predict_integer(vector)
        c_score_q, c_flag = _run_c_fixed(compiled_fixed_harness, vector)
        assert c_score_q == py["score_q"]
        assert c_flag == py["is_anomaly"]

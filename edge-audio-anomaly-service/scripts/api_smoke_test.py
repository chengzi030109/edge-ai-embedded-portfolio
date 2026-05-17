from __future__ import annotations

"""Start the FastAPI service and exercise the embedded-Linux API contract."""

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.config import load_config
from edge_audio.features import load_feature_rows
from edge_audio.model import AudioCentroidModel
from edge_audio.streaming import collect_window_feature_vectors
from edge_audio.synth import generate_demo_wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run API smoke test against a temporary local uvicorn service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--timeout", type=float, default=15.0)
    return parser.parse_args()


def ensure_demo_model_and_wav() -> Path:
    """Create deterministic WAV/model artifacts if this is a fresh checkout."""

    cfg = load_config(ROOT / "configs/default.toml")
    generate_demo_wavs(cfg.data_dir, cfg.sample_rate_hz, cfg.clip_seconds)
    rows = load_feature_rows(cfg.data_dir)
    if not cfg.model_path.exists():
        vectors = collect_window_feature_vectors(rows, cfg.window_seconds, cfg.hop_seconds)
        AudioCentroidModel.train(vectors).save(cfg.model_path)
    return Path(rows[0]["path"])


def ensure_api_dependencies() -> None:
    """Fail early with install guidance when optional API packages are absent."""

    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        import multipart  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            "API smoke test requires optional API dependencies. "
            "Install with: pip install -e .[api] or pip install -r requirements.txt. "
            f"Original error: {exc}"
        ) from exc


def main() -> None:
    args = parse_args()
    ensure_api_dependencies()
    sample_wav = ensure_demo_model_and_wav()
    env = dict(**__import__("os").environ, PYTHONPATH=str(ROOT / "src"))
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "edge_audio.api:create_app",
        "--factory",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--log-level",
        "warning",
    ]
    process = subprocess.Popen(command, cwd=ROOT, env=env)
    base_url = f"http://{args.host}:{args.port}"
    try:
        wait_for_health(base_url, args.timeout)
        upload = post_multipart_upload(base_url + "/api/v1/audio/upload", sample_wav)
        metrics = http_get(base_url + "/metrics").decode("utf-8")
        events = json.loads(http_get(base_url + "/api/v1/audio/events?limit=1").decode("utf-8"))
        ack = post_json(base_url + "/api/v1/audio/events/ack", {"event_ids": [events[0]["id"]], "ack": True})
        print("api smoke test passed")
        print(f"upload windows: {upload['count']}")
        print(f"metrics contains events: {'edge_audio_events_total' in metrics}")
        print(f"ack updated: {ack['updated']}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def wait_for_health(base_url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            payload = json.loads(http_get(base_url + "/healthz").decode("utf-8"))
            if payload.get("status") == "ok":
                return
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"service did not become healthy: {last_error}")


def http_get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.read()


def post_json(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=5) as response:
        return json.loads(response.read().decode("utf-8"))


def post_multipart_upload(url: str, wav_path: Path) -> dict:
    """POST one WAV file using a tiny stdlib-only multipart encoder."""

    boundary = f"edge-audio-boundary-{int(time.time() * 1000)}"
    wav_bytes = wav_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="label"\r\n\r\n',
            b"normal\r\n",
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="file"; filename="{wav_path.name}"\r\n'.encode(),
            b"Content-Type: audio/wav\r\n\r\n",
            wav_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        ]
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(exc.read().decode("utf-8")) from exc


if __name__ == "__main__":
    main()

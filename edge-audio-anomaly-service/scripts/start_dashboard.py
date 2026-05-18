from __future__ import annotations

"""Start the API/dashboard service after preparing demo artifacts.

The Docker and embedded-Linux service paths both need the same first-run
bootstrap: make sure deterministic demo WAV files exist, train a lightweight
centroid model if no model has been deployed yet, and then hand off to uvicorn.
Keeping that bootstrap in Python makes the container command easy to read and
keeps Windows, Linux, and Docker behavior aligned.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from edge_audio.config import load_config
from edge_audio.features import load_feature_rows
from edge_audio.model import AudioCentroidModel
from edge_audio.storage import connect, init_db
from edge_audio.streaming import collect_window_feature_vectors
from edge_audio.synth import generate_demo_wavs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare demo artifacts and start the dashboard service")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--backend", choices=["centroid", "onnx"], default=None)
    return parser.parse_args()


def ensure_demo_state():
    """Create local data/model/database files when the checkout is fresh."""

    cfg = load_config(ROOT / "configs/default.toml")
    generate_demo_wavs(cfg.data_dir, cfg.sample_rate_hz, cfg.clip_seconds)
    rows = load_feature_rows(cfg.data_dir)

    # The dashboard can start from an empty repository or an empty Docker
    # volume. Training this tiny one-class model on normal demo windows gives
    # the API a valid backend before the first upload arrives.
    if not cfg.model_path.exists():
        vectors = collect_window_feature_vectors(rows, cfg.window_seconds, cfg.hop_seconds)
        AudioCentroidModel.train(vectors).save(cfg.model_path)

    conn = connect(cfg.database_path)
    init_db(conn)
    conn.close()
    return cfg


def main() -> None:
    args = parse_args()
    cfg = ensure_demo_state()

    import uvicorn
    from edge_audio.api import create_app

    # The API factory keeps model/backend paths configurable. For the portfolio
    # container we use the default config files and expose the same dashboard
    # that systemd would host on an embedded Linux board.
    app = create_app(
        model_path=cfg.model_path,
        database_path=cfg.database_path,
        backend=args.backend or cfg.model_backend,
        onnx_model_path=cfg.onnx_model_path,
        safe_roots=[cfg.data_dir],
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()

from __future__ import annotations

import json

from edge_gateway.api import ROUTES
from edge_gateway.config import load_config
from edge_gateway.replay import replay_jsonl
from edge_gateway.report import render_markdown
from edge_gateway.storage import connect, init_db, ingest_telemetry, list_devices, summary


def test_config_loads_default_paths():
    cfg = load_config("configs/default.toml")
    assert cfg.device_id == "sim-node-001"
    assert cfg.database_path.name == "gateway.db"


def test_storage_ingest_and_summary(tmp_path):
    conn = connect(tmp_path / "gateway.db")
    init_db(conn)
    ingest_telemetry(
        conn,
        {"seq": 1, "timestamp": 1.0, "score": 7.0, "threshold": 6.0, "is_alarm": True, "alarm_state": "alarm"},
        "dev-a",
    )
    assert list_devices(conn)[0]["device_id"] == "dev-a"
    assert summary(conn)["alarm_event_count"] == 1


def test_jsonl_replay_and_report(tmp_path):
    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps({"seq": 1, "timestamp": 1.0, "score": 1.0, "threshold": 6.0, "alarm_state": "normal"}) + "\n",
        encoding="utf-8",
    )
    conn = connect(tmp_path / "gateway.db")
    init_db(conn)
    assert replay_jsonl(conn, telemetry, "dev-a") == 1
    assert "Telemetry rows" in render_markdown(conn)


def test_api_route_contract_is_declared():
    assert "POST /api/v1/telemetry" in ROUTES
    assert "GET /api/v1/summary" in ROUTES


from __future__ import annotations

"""Optional FastAPI routes for audio analysis."""

import time
from pathlib import Path
from threading import Lock

from .alarm import AlarmDebouncer
from .backends import load_backend
from .features import extract_features, read_wav
from .storage import connect, init_db, insert_events, list_events, mark_events_uploaded, summary
from .streaming import analyze_wav_windows

ROUTES = [
    "GET /",
    "GET /dashboard",
    "POST /api/v1/audio/analyze",
    "POST /api/v1/audio/analyze-windowed",
    "POST /api/v1/audio/upload",
    "POST /api/v1/audio/events/ack",
    "GET /api/v1/audio/events",
    "GET /api/v1/audio/summary",
    "GET /healthz",
    "GET /metrics",
]


def create_app(
    model_path: str | Path = "artifacts/audio_model.json",
    database_path: str | Path = "data/audio_events.db",
    backend: str = "centroid",
    onnx_model_path: str | Path = "artifacts/audio_model.onnx",
):
    """Create the optional FastAPI application.

    FastAPI is deliberately optional because the core portfolio demo should run
    on a plain Python environment. When FastAPI is installed, this factory turns
    the same feature/model code into a Linux service endpoint. Events are stored
    in SQLite instead of only an in-memory list so the application behaves like
    an edge gateway that can keep local evidence during network outages.
    """

    try:
        from fastapi import FastAPI, File, UploadFile
        from fastapi.responses import HTMLResponse, PlainTextResponse
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("FastAPI is optional; install requirements.txt or pip install -e .[api] to run the API server") from exc
    globals()["UploadFile"] = UploadFile

    app = FastAPI(title="Edge Audio Anomaly Service")
    conn = connect(database_path)
    init_db(conn)
    db_lock = Lock()
    upload_dir = Path(database_path).resolve().parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    def current_model():
        """Load the configured backend on demand so model files can be replaced."""

        return load_backend(backend, centroid_model_path=model_path, onnx_model_path=onnx_model_path)

    @app.get("/", response_class=HTMLResponse)
    @app.get("/dashboard", response_class=HTMLResponse)
    def dashboard():
        """Serve a small browser dashboard for local operator demos.

        The page is intentionally plain HTML, CSS, and browser fetch() calls.
        That keeps embedded Linux deployment simple: no Node.js runtime, no
        frontend build step, and no second service to supervise with systemd.
        """

        return DASHBOARD_HTML

    @app.post("/api/v1/audio/analyze")
    def analyze(payload: dict):
        """Analyze a complete WAV file as one clip-level request.

        The windowed endpoint is closer to the embedded runtime, but this
        simpler endpoint is useful for API smoke tests and one-off uploads from
        a factory operator or a batch script.
        """

        model = current_model()
        start = time.perf_counter()
        samples, sr = read_wav(payload["path"])
        features = extract_features(samples, sr)
        feature_ms = (time.perf_counter() - start) * 1000.0
        infer_start = time.perf_counter()
        result = model.predict(features)
        inference_ms = (time.perf_counter() - infer_start) * 1000.0
        event = {
            "source": payload["path"],
            "label": str(payload.get("label", "unknown")),
            "window_index": 0,
            "start_s": 0.0,
            "end_s": float(len(samples) / sr),
            "score": result["score"],
            "threshold": result["threshold"],
            "is_anomaly_raw": result["is_anomaly"],
            "is_anomaly": result["is_anomaly"],
            "is_alarm": result["is_anomaly"],
            "alarm_state": "alarm" if result["is_anomaly"] else "normal",
            "alarm_bad_streak": 1 if result["is_anomaly"] else 0,
            "alarm_good_streak": 0 if result["is_anomaly"] else 1,
            "feature_ms": feature_ms,
            "inference_ms": inference_ms,
            "clip_path": "",
            "features": {name: float(value) for name, value in zip(model.feature_names, features, strict=False)},
        }
        with db_lock:
            insert_events(conn, [event])
        return event

    @app.post("/api/v1/audio/analyze-windowed")
    def analyze_windowed(payload: dict):
        """Analyze a WAV file as stream windows and persist every event."""

        model = current_model()
        rows = analyze_wav_windows(
            payload["path"],
            str(payload.get("label", "unknown")),
            model,
            float(payload.get("window_seconds", 0.25)),
            float(payload.get("hop_seconds", 0.125)),
            debouncer=AlarmDebouncer(
                on_count=int(payload.get("alarm_on_count", 3)),
                off_count=int(payload.get("alarm_off_count", 5)),
            ),
        )
        with db_lock:
            insert_events(conn, rows)
        return {"events": rows, "count": len(rows)}

    @app.post("/api/v1/audio/upload")
    async def upload_audio(
        file: UploadFile = File(...),
        label: str = "unknown",
        window_seconds: float = 0.25,
        hop_seconds: float = 0.125,
        alarm_on_count: int = 3,
        alarm_off_count: int = 5,
    ):
        """Upload a WAV file, persist it locally, analyze windows, and buffer events.

        On a board this mimics a file-drop or remote operator upload workflow.
        The uploaded bytes are saved first so later debugging can replay the
        exact audio that produced each SQLite event.
        """

        safe_name = Path(file.filename or "upload.wav").name
        saved_path = upload_dir / f"{int(time.time() * 1000)}_{safe_name}"
        saved_path.write_bytes(await file.read())
        model = current_model()
        rows = analyze_wav_windows(
            saved_path,
            str(label),
            model,
            float(window_seconds),
            float(hop_seconds),
            debouncer=AlarmDebouncer(on_count=int(alarm_on_count), off_count=int(alarm_off_count)),
        )
        with db_lock:
            insert_events(conn, rows)
        return {
            "path": str(saved_path),
            "count": len(rows),
            "raw_anomalies": sum(1 for row in rows if row["is_anomaly_raw"]),
            "alarms": sum(1 for row in rows if row["is_alarm"]),
            "events": rows,
        }

    @app.post("/api/v1/audio/events/ack")
    def ack_events(payload: dict):
        """Mark buffered events as uploaded and optionally acknowledged."""

        event_ids = [int(event_id) for event_id in payload.get("event_ids", [])]
        with db_lock:
            changed = mark_events_uploaded(conn, event_ids, ack=bool(payload.get("ack", True)))
        return {"updated": changed, "ack": bool(payload.get("ack", True))}

    @app.get("/api/v1/audio/events")
    def get_events(limit: int = 50):
        with db_lock:
            return list_events(conn, limit=limit)

    @app.get("/api/v1/audio/summary")
    def get_summary():
        with db_lock:
            return summary(conn)

    @app.get("/healthz")
    def healthz():
        return {
            "status": "ok",
            "backend": backend,
            "model_exists": Path(model_path).exists() if backend == "centroid" else Path(onnx_model_path).exists(),
            "database": str(database_path),
        }

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics():
        with db_lock:
            stats = summary(conn)
        lines = [
            "# HELP edge_audio_events_total Total audio window events buffered locally.",
            "# TYPE edge_audio_events_total counter",
            f"edge_audio_events_total {stats['event_count']}",
            "# HELP edge_audio_raw_anomalies_total Raw anomaly windows emitted by the model.",
            "# TYPE edge_audio_raw_anomalies_total counter",
            f"edge_audio_raw_anomalies_total {stats['raw_anomaly_count']}",
            "# HELP edge_audio_alarm_windows_total Debounced alarm windows.",
            "# TYPE edge_audio_alarm_windows_total counter",
            f"edge_audio_alarm_windows_total {stats['alarm_count']}",
            "# HELP edge_audio_pending_upload_events Locally buffered events not marked uploaded.",
            "# TYPE edge_audio_pending_upload_events gauge",
            f"edge_audio_pending_upload_events {stats['pending_upload_count']}",
            "# HELP edge_audio_inference_ms_avg Average model inference latency in milliseconds.",
            "# TYPE edge_audio_inference_ms_avg gauge",
            f"edge_audio_inference_ms_avg {stats['inference_ms_avg']:.6f}",
            "",
        ]
        return "\n".join(lines)

    return app


DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Edge Audio Anomaly Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #667085;
      --line: #d8dee9;
      --accent: #0f766e;
      --alarm: #b42318;
      --ok: #087443;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      padding: 20px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: #fff;
    }
    h1 {
      margin: 0;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .sub {
      margin-top: 4px;
      color: var(--muted);
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 20px;
      display: grid;
      gap: 16px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }
    .card, section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
    }
    .value {
      margin-top: 4px;
      font-size: 28px;
      font-weight: 750;
    }
    .ok { color: var(--ok); }
    .alarm { color: var(--alarm); }
    .grid {
      display: grid;
      grid-template-columns: minmax(260px, 0.8fr) minmax(0, 1.2fr);
      gap: 16px;
    }
    @media (max-width: 820px) {
      .grid { grid-template-columns: 1fr; }
    }
    h2 {
      margin: 0 0 12px;
      font-size: 16px;
    }
    form {
      display: grid;
      gap: 10px;
    }
    input, button {
      width: 100%;
      min-height: 38px;
      border-radius: 6px;
      border: 1px solid var(--line);
      padding: 8px 10px;
      font: inherit;
    }
    button {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
      cursor: pointer;
      font-weight: 650;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      padding: 8px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
    }
    th {
      color: var(--muted);
      font-weight: 650;
    }
    .status {
      margin-top: 8px;
      color: var(--muted);
      min-height: 20px;
    }
    pre {
      margin: 0;
      overflow: auto;
      max-height: 240px;
      background: #0f172a;
      color: #e5edf7;
      border-radius: 6px;
      padding: 12px;
      font-size: 12px;
    }
  </style>
</head>
<body>
  <header>
    <h1>Edge Audio Anomaly Dashboard</h1>
    <div class="sub">Local embedded-Linux service view for WAV upload, alarm state, SQLite events, and Prometheus-style metrics.</div>
  </header>
  <main>
    <div class="stats">
      <div class="card"><div class="label">Buffered Events</div><div id="events" class="value">0</div></div>
      <div class="card"><div class="label">Raw Anomalies</div><div id="raw" class="value">0</div></div>
      <div class="card"><div class="label">Alarm Windows</div><div id="alarms" class="value">0</div></div>
      <div class="card"><div class="label">Pending Upload</div><div id="pending" class="value">0</div></div>
    </div>
    <div class="grid">
      <section>
        <h2>Upload WAV</h2>
        <form id="upload-form">
          <input id="file" type="file" accept=".wav,audio/wav" required />
          <input id="label" type="text" value="unknown" aria-label="label" />
          <button type="submit">Analyze Upload</button>
        </form>
        <div id="upload-status" class="status"></div>
      </section>
      <section>
        <h2>Metrics</h2>
        <pre id="metrics">loading...</pre>
      </section>
    </div>
    <section>
      <h2>Recent Events</h2>
      <table>
        <thead>
          <tr>
            <th>ID</th>
            <th>State</th>
            <th>Score</th>
            <th>Threshold</th>
            <th>Uploaded</th>
            <th>Ack</th>
            <th>Source</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </section>
  </main>
  <script>
    async function refresh() {
      const [summary, events, metrics] = await Promise.all([
        fetch('/api/v1/audio/summary').then(r => r.json()),
        fetch('/api/v1/audio/events?limit=12').then(r => r.json()),
        fetch('/metrics').then(r => r.text())
      ]);
      document.getElementById('events').textContent = summary.event_count;
      document.getElementById('raw').textContent = summary.raw_anomaly_count;
      const alarms = document.getElementById('alarms');
      alarms.textContent = summary.alarm_count;
      alarms.className = 'value ' + (summary.alarm_count > 0 ? 'alarm' : 'ok');
      document.getElementById('pending').textContent = summary.pending_upload_count;
      document.getElementById('metrics').textContent = metrics.trim();
      const tbody = document.getElementById('rows');
      tbody.innerHTML = '';
      for (const event of events) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
          <td>${event.id}</td>
          <td>${event.alarm_state}</td>
          <td>${Number(event.score).toFixed(3)}</td>
          <td>${Number(event.threshold).toFixed(3)}</td>
          <td>${event.uploaded}</td>
          <td>${event.ack}</td>
          <td>${event.source}</td>`;
        tbody.appendChild(tr);
      }
    }

    document.getElementById('upload-form').addEventListener('submit', async (ev) => {
      ev.preventDefault();
      const file = document.getElementById('file').files[0];
      const label = document.getElementById('label').value || 'unknown';
      const body = new FormData();
      body.append('file', file);
      body.append('label', label);
      const status = document.getElementById('upload-status');
      status.textContent = 'uploading and analyzing...';
      try {
        const response = await fetch('/api/v1/audio/upload', { method: 'POST', body });
        if (!response.ok) {
          const text = await response.text();
          throw new Error(text || `HTTP ${response.status}`);
        }
        const payload = await response.json();
        status.textContent = `analyzed ${payload.count || 0} windows; alarms ${payload.alarms || 0}`;
        await refresh();
      } catch (err) {
        status.textContent = `upload failed: ${err.message || err}`;
      }
    });

    refresh();
    setInterval(refresh, 4000);
  </script>
</body>
</html>
"""

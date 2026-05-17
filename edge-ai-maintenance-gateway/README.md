# Edge AI Maintenance Gateway

Embedded Linux application-layer gateway for the TinyML predictive-maintenance
node. It runs without hardware by replaying JSONL telemetry from
`tinyml-predictive-maintenance`, then stores data in SQLite and generates a
static dashboard.

## Quick Start

```powershell
cd E:\linux\edge-ai-maintenance-gateway
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe scripts\run_gateway_demo.py
..\tinyml-predictive-maintenance\.venv\Scripts\python.exe -m pytest -q
```

Outputs:

- `data/gateway.db`
- `reports/gateway_report.md`
- `reports/dashboard.html`

## API Contract

- `POST /api/v1/telemetry`
- `GET /api/v1/devices`
- `GET /api/v1/telemetry?device_id=&limit=`
- `GET /api/v1/alarms`
- `GET /api/v1/summary`

FastAPI is optional. Install `requirements.txt` when you want to run the HTTP
server; the local demo and tests do not require it.

## Embedded Linux Notes

- JSONL replay represents UART/MQTT input during hardware-free development.
- SQLite gives local buffering when an edge device is offline.
- `systemd/edge-ai-maintenance-gateway.service` shows the deployment shape.
- Logs can be handled by journald or logrotate in a real Linux image.


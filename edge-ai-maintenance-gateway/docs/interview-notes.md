# Interview Notes

This project is an embedded Linux application-layer gateway. It receives
telemetry from a TinyML maintenance node, stores it locally in SQLite, exposes
API handlers, and generates a local dashboard/report.

Key talking points:

- JSONL replay stands in for UART/MQTT input while hardware is unavailable.
- SQLite is used as an edge buffer when the network or cloud is unavailable.
- API handlers and reports share the same storage layer.
- The gateway is deployable as a systemd service on a Linux board.

Demo command:

```powershell
python scripts/run_gateway_demo.py
```


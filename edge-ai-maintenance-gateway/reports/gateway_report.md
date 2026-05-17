# Edge AI Maintenance Gateway Report

- Telemetry rows: `25`
- Alarm events: `13`
- Avg feature+inference latency: `0.4011 ms`

## Devices

| Device | Last seen | Last alarm state |
|---|---:|---|
| sim-node-001 | 1779024586.579 | alarm |

## Recent Alarms

| Seq | State | Score | Threshold |
|---:|---|---:|---:|
| 24 | alarm | 43.229 | 6.089 |
| 23 | alarm | 56.908 | 6.089 |
| 22 | alarm | 52.880 | 6.089 |
| 21 | alarm | 55.382 | 6.089 |
| 20 | alarm | 19.442 | 6.089 |
| 19 | alarm | 140.363 | 6.089 |
| 18 | alarm | 123.996 | 6.089 |
| 17 | alarm | 124.117 | 6.089 |
| 16 | alarm | 50.924 | 6.089 |
| 15 | alarm | 47.968 | 6.089 |

## Linux Application Notes

This gateway uses SQLite for local buffering and exposes the same data through
API handlers. MQTT/UART ingestion can reuse `ingest_telemetry()` without
changing dashboard or report logic.

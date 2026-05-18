# Portfolio Roadmap

This roadmap keeps the repository focused on internship value: projects should
be runnable on a laptop, explainable in interviews, and close to embedded
deployment patterns.

## Current Portfolio Shape

| Area | Project | Current Value |
|---|---|---|
| MCU / TinyML | `tinyml-predictive-maintenance` | Sensor-window simulation, anomaly scoring, fixed-point reports, C export, PHM2008 path. |
| Embedded Linux audio | `edge-audio-anomaly-service` | Best showcase: replay/upload, SQLite, ONNX backend, FastAPI contracts, systemd, metrics. |
| Embedded Linux gateway | `edge-ai-maintenance-gateway` | Telemetry ingest, local persistence, summary/dashboard shape. |
| Embedded Linux vision | `edge-vision-inspection` | Batch inspection, synthetic defects, annotated reports. |
| Measurement | `edgebench` | Simple edge model benchmark and report utility. |

## Next Priorities

1. **Make GitHub easy to scan**
   - Keep the root README short and visual.
   - Keep each project README runnable in under 10 minutes.
   - Prefer stable report links over long explanations.

2. **Harden CI**
   - Keep `edge-audio-anomaly-service` tests green on Ubuntu.
   - Keep TinyML lite tests separate from heavier torch/ONNX paths.
   - Avoid CI steps that require large public datasets.

3. **Improve real-data credibility**
   - Download real MIMII or ToyADMOS data outside Git.
   - Run `scripts/evaluate_public_audio_dataset.py --data-root ...`.
   - Add real result screenshots or report excerpts only after verifying data source and split.

4. **Service deployment polish**
   - Run the audio API smoke test after FastAPI dependencies are installed.
   - Add a short terminal recording or screenshot of `/metrics`.
   - Keep systemd/logrotate docs aligned with the API entrypoint.

## Nice-To-Have Later

- Add a minimal Streamlit or static dashboard for the audio service.
- Add one real webcam/USB-camera capture adapter for vision when hardware is available.
- Add MQTT upload simulation for gateway and audio event batches.
- Add a one-page PDF resume appendix generated from `docs/interview-cheatsheet.md`.

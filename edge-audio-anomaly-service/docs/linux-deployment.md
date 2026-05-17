# Embedded Linux Deployment Notes

This project is designed so the laptop demo and the board deployment share the
same middle of the pipeline: audio windows become feature vectors, feature
vectors become anomaly scores, and anomaly scores become events in SQLite.

## Current Laptop Prototype

- `scripts/run_audio_demo.py` generates deterministic WAV clips, trains a small
  one-class centroid model from normal windows, replays every WAV as overlapping
  stream windows, stores events in SQLite, saves anomaly snippets, and writes a
  Markdown report.
- `src/edge_audio/features.py` owns the audio contract: mono 16-bit PCM WAV in,
  nine lightweight features out.
- `src/edge_audio/streaming.py` owns the runtime contract: fixed-length windows
  with start/end timestamps, feature latency, inference latency, and optional
  anomaly clip extraction.
- `src/edge_audio/alarm.py` owns the debounce contract: raw model decisions are
  converted into `normal`, `pending`, `alarm`, and `recovering` device states.
- `src/edge_audio/api.py` is optional. When FastAPI is installed, it exposes the
  same analyzer through HTTP and persists events through the same SQLite schema.
- `src/edge_audio/backends.py` owns the inference backend boundary. The board
  can start with the Python centroid backend and later switch to ONNX Runtime.
- `scripts/query_audio_events.py` is the SSH-friendly operator tool for checking
  the local event buffer without starting the API server.

## Board Input Boundary

The only part that should change on an embedded Linux board is the input source.
The rest of the code should remain stable.

| Laptop demo source | Board replacement | Notes |
|---|---|---|
| WAV folder replay | ALSA microphone capture | Use the same sample rate and window size where possible. |
| WAV folder replay | UDP audio packets | Buffer packets until one full window is available. |
| Synthetic WAV generation | Real machine recordings | Keep a small labeled normal/anomaly folder for regression tests. |
| Local report run | Long-running API service | Use `systemd/edge-audio-anomaly-service.service`. |

## Runtime Resource Budget

The default configuration is intentionally small enough to explain in an
embedded interview.

| Item | Default | Why it matters |
|---|---:|---|
| Audio window | 0.25 s at 16 kHz | 4000 samples, about 8 KB as int16 PCM. |
| Hop size | 0.125 s | 50 percent overlap, eight decisions per second. |
| Feature vector | 9 float32 values | About 36 bytes before JSON/event overhead. |
| Centroid model | mean + scale + threshold | About 76 bytes as float32 parameters. |
| Alarm debounce | two counters + state | Converts raw spikes into stable device alarms. |
| Event store | SQLite | Survives process restart and network outage. |

## Service Process Shape

For a board image, install dependencies into a venv, copy the project to
`/opt/edge-audio-anomaly-service`, generate or deploy `artifacts/audio_model.json`,
then enable the service file.

```bash
sudo cp systemd/edge-audio-anomaly-service.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now edge-audio-anomaly-service
sudo journalctl -u edge-audio-anomaly-service -f
```

When debugging over SSH, query the local buffer directly:

```bash
python3 scripts/query_audio_events.py --limit 20
python3 scripts/query_audio_events.py --json --limit 5
```

Recommended production split:

| Worker | Responsibility |
|---|---|
| Capture worker | Reads ALSA/UDP bytes and produces fixed-size PCM windows. |
| Feature worker | Computes RMS, ZCR, spectrum centroid, flatness, and band energy. |
| Inference worker | Runs centroid or ONNX Runtime scoring. |
| Alarm worker | Applies consecutive-window debounce and emits stable alarm state. |
| Storage worker | Writes window events and anomaly snippets locally. |
| API/upload worker | Serves recent events or uploads batches when the network is healthy. |

## Next Upgrade Path

- Replace WAV replay with an ALSA reader while keeping `extract_features` and
  `AudioCentroidModel.predict` unchanged.
- Export `artifacts/audio_model.onnx`, install ONNX Runtime on the board, and
  start the same service with the ONNX backend.
- Add log rotation for `journalctl` and a retention policy for
  `reports/anomaly_clips/`.
- Add an upload job that reads SQLite rows by ID and marks them as uploaded
  after the cloud side acknowledges the batch.

## ONNX Runtime Placement

ONNX Runtime belongs only inside the inference worker. The board still captures
PCM windows, extracts the same nine features, applies the same alarm debounce,
and writes the same SQLite event rows. This keeps deployment risk low because a
model backend swap does not rewrite the Linux service around it.

# Edge Audio Anomaly Report

- Windows analyzed: `91`
- Raw anomaly windows: `35`
- Debounced alarm windows: `33`
- Saved anomaly clips: `35`
- Raw model precision / recall / F1: `1.000` / `1.000` / `1.000`
- Avg feature latency: `0.1532 ms`
- Avg inference latency: `0.0051 ms`
- Model backend: `centroid`
- Model path: `audio_model.json`
- Figure: `audio_score_curve.png`

![Audio score curve](audio_score_curve.png)

## Confusion Matrix

| TP | TN | FP | FN |
|---:|---:|---:|---:|
| 35 | 56 | 0 | 0 |

The demo data is synthetic, so these metrics validate the replay,
windowing, feature, scoring, alarm debounce, storage, and report pipeline. Field accuracy
should be measured again with real machine recordings from the target
factory environment.

## Recent Window Events

| File | Window | Label | Score | Threshold | Raw | Alarm | State | Clip |
|---|---:|---|---:|---:|---|---|---|---|
| normal_00.wav | 0 | normal | 0.848 | 9.586 | False | False | normal |  |
| normal_00.wav | 1 | normal | 0.801 | 9.586 | False | False | normal |  |
| normal_00.wav | 2 | normal | 0.787 | 9.586 | False | False | normal |  |
| normal_00.wav | 3 | normal | 1.391 | 9.586 | False | False | normal |  |
| normal_00.wav | 4 | normal | 1.043 | 9.586 | False | False | normal |  |
| normal_00.wav | 5 | normal | 1.194 | 9.586 | False | False | normal |  |
| normal_00.wav | 6 | normal | 1.198 | 9.586 | False | False | normal |  |
| normal_01.wav | 0 | normal | 2.770 | 9.586 | False | False | normal |  |
| normal_01.wav | 1 | normal | 2.675 | 9.586 | False | False | normal |  |
| normal_01.wav | 2 | normal | 2.530 | 9.586 | False | False | normal |  |
| normal_01.wav | 3 | normal | 2.460 | 9.586 | False | False | normal |  |
| normal_01.wav | 4 | normal | 2.583 | 9.586 | False | False | normal |  |
| normal_01.wav | 5 | normal | 2.610 | 9.586 | False | False | normal |  |
| normal_01.wav | 6 | normal | 3.351 | 9.586 | False | False | normal |  |
| normal_02.wav | 0 | normal | 2.578 | 9.586 | False | False | normal |  |
| normal_02.wav | 1 | normal | 2.200 | 9.586 | False | False | normal |  |
| normal_02.wav | 2 | normal | 2.150 | 9.586 | False | False | normal |  |
| normal_02.wav | 3 | normal | 2.248 | 9.586 | False | False | normal |  |
| normal_02.wav | 4 | normal | 2.280 | 9.586 | False | False | normal |  |
| normal_02.wav | 5 | normal | 2.232 | 9.586 | False | False | normal |  |
| normal_02.wav | 6 | normal | 2.312 | 9.586 | False | False | normal |  |
| normal_03.wav | 0 | normal | 3.217 | 9.586 | False | False | normal |  |
| normal_03.wav | 1 | normal | 1.823 | 9.586 | False | False | normal |  |
| normal_03.wav | 2 | normal | 1.818 | 9.586 | False | False | normal |  |
| normal_03.wav | 3 | normal | 1.592 | 9.586 | False | False | normal |  |
| normal_03.wav | 4 | normal | 1.935 | 9.586 | False | False | normal |  |
| normal_03.wav | 5 | normal | 1.841 | 9.586 | False | False | normal |  |
| normal_03.wav | 6 | normal | 1.850 | 9.586 | False | False | normal |  |
| normal_04.wav | 0 | normal | 6.224 | 9.586 | False | False | normal |  |
| normal_04.wav | 1 | normal | 6.502 | 9.586 | False | False | normal |  |
| normal_04.wav | 2 | normal | 6.563 | 9.586 | False | False | normal |  |
| normal_04.wav | 3 | normal | 6.464 | 9.586 | False | False | normal |  |
| normal_04.wav | 4 | normal | 6.364 | 9.586 | False | False | normal |  |
| normal_04.wav | 5 | normal | 6.879 | 9.586 | False | False | normal |  |
| normal_04.wav | 6 | normal | 7.181 | 9.586 | False | False | normal |  |
| normal_05.wav | 0 | normal | 2.675 | 9.586 | False | False | normal |  |
| normal_05.wav | 1 | normal | 2.494 | 9.586 | False | False | normal |  |
| normal_05.wav | 2 | normal | 2.633 | 9.586 | False | False | normal |  |
| normal_05.wav | 3 | normal | 2.787 | 9.586 | False | False | normal |  |
| normal_05.wav | 4 | normal | 2.967 | 9.586 | False | False | normal |  |
| normal_05.wav | 5 | normal | 2.645 | 9.586 | False | False | normal |  |
| normal_05.wav | 6 | normal | 2.449 | 9.586 | False | False | normal |  |
| normal_06.wav | 0 | normal | 1.482 | 9.586 | False | False | normal |  |
| normal_06.wav | 1 | normal | 1.424 | 9.586 | False | False | normal |  |
| normal_06.wav | 2 | normal | 1.510 | 9.586 | False | False | normal |  |
| normal_06.wav | 3 | normal | 1.561 | 9.586 | False | False | normal |  |
| normal_06.wav | 4 | normal | 1.555 | 9.586 | False | False | normal |  |
| normal_06.wav | 5 | normal | 1.478 | 9.586 | False | False | normal |  |
| normal_06.wav | 6 | normal | 1.501 | 9.586 | False | False | normal |  |
| normal_07.wav | 0 | normal | 1.489 | 9.586 | False | False | normal |  |
| normal_07.wav | 1 | normal | 0.554 | 9.586 | False | False | normal |  |
| normal_07.wav | 2 | normal | 0.637 | 9.586 | False | False | normal |  |
| normal_07.wav | 3 | normal | 0.941 | 9.586 | False | False | normal |  |
| normal_07.wav | 4 | normal | 0.844 | 9.586 | False | False | normal |  |
| normal_07.wav | 5 | normal | 0.972 | 9.586 | False | False | normal |  |
| normal_07.wav | 6 | normal | 0.887 | 9.586 | False | False | normal |  |
| anomaly_00.wav | 0 | anomaly | 44.754 | 9.586 | True | False | pending | anomaly_00_win000.wav |
| anomaly_00.wav | 1 | anomaly | 43.789 | 9.586 | True | False | pending | anomaly_00_win001.wav |
| anomaly_00.wav | 2 | anomaly | 44.926 | 9.586 | True | True | alarm | anomaly_00_win002.wav |
| anomaly_00.wav | 3 | anomaly | 44.751 | 9.586 | True | True | alarm | anomaly_00_win003.wav |

## Resource Budget

- Feature vector: 9 float32 values, about 36 bytes.
- Feature names: `rms, peak, crest_factor, zcr, spectral_centroid_hz, spectral_flatness, low_band, mid_band, high_band`
- Default stream window: 0.25 s at 16 kHz, about 8 KB as int16 PCM.
- Model parameters: mean + scale + threshold, about 76 bytes as float32.
- Alarm debounce state: two counters and one boolean, suitable for a small service task.
- SQLite buffers events locally so the service can tolerate network loss.

## Linux Application Notes

WAV replay stands in for microphone or UDP audio input. The feature and
model boundary can stay stable when the input source becomes ALSA, PulseAudio,
or an embedded recorder process.

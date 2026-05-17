# Evaluation Report

- Sample rate: `1600` Hz
- Window size: `256` samples
- Windows per state: `40`
- Threshold: `6.0886`

## Binary Anomaly Metrics

| Metric | Value |
|---|---:|
| Accuracy | 0.9750 |
| Precision | 1.0000 |
| Recall | 0.9667 |
| F1 | 0.9831 |
| False positive rate | 0.0000 |
| False negative rate | 0.0333 |

## Per-State Detection

| State | Windows | Detected as anomaly | Detection rate |
|---|---:|---:|---:|
| normal | 40 | 0 | 0.0000 |
| imbalance | 40 | 40 | 1.0000 |
| rubbing | 40 | 36 | 0.9000 |
| bearing | 40 | 40 | 1.0000 |

## Latency

| Stage | Avg ms | P95 ms |
|---|---:|---:|
| Feature extraction | 0.087008 | 0.108850 |
| Inference | 0.004657 | 0.008305 |

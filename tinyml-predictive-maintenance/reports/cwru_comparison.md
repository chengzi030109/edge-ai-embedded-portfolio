# Model Comparison on CWRU

- Source: `cwru`
- Window size: `1024` samples @ 12000 Hz
- Train (normal) windows: `280`
- Test windows: `832` (normal=120, faulty=712)

## Detection Metrics

| Model | Accuracy | Precision | Recall | F1 | FPR | Avg latency (ms) | Size (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CentroidAnomalyDetector | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0034 | 641 |
| IsolationForest | 0.9712 | 1.0000 | 0.9663 | 0.9829 | 0.0000 | 5.0050 | 1227444 |
| OneClassSVM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.0624 | 2600 |
| LocalOutlierFactor | 0.9988 | 0.9986 | 1.0000 | 0.9993 | 0.0083 | 0.3167 | 64762 |
| Autoencoder1D | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | 0.1450 | 12031 |

## Per-Label Detection Rate

| Model | normal | inner | outer | ball |
|---|---:|---:|---:|---:|
| CentroidAnomalyDetector | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| IsolationForest | 0.0000 | 1.0000 | 1.0000 | 0.8987 |
| OneClassSVM | 0.0000 | 1.0000 | 1.0000 | 1.0000 |
| LocalOutlierFactor | 0.0083 | 1.0000 | 1.0000 | 1.0000 |
| Autoencoder1D | 0.0000 | 1.0000 | 1.0000 | 1.0000 |

# PHM2008 / C-MAPSS Degradation Comparison

- Source file: `data\phm2008_sample\train_FD001.txt`
- Window size: `30` cycles
- Hop: `10` cycles
- Features: `32`
- Train windows: `25`  Test windows: `47`

PHM2008 is harder than CWRU for this project because the signal is
multivariate, multi-unit, and gradually degrading rather than a seeded
bearing fault with a large spectral separation.

| Model | Accuracy | Precision | Recall | F1 | FPR | Avg latency (ms) | Size (bytes) |
|---|---:|---:|---:|---:|---:|---:|---:|
| CentroidAnomalyDetector | 0.9362 | 0.9231 | 1.0000 | 0.9600 | 0.2727 | 0.0032 | 260 |

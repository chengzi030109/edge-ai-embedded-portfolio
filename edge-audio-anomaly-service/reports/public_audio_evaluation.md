# Public Audio Dataset Evaluation

- Dataset root: `data\public_audio_sample`
- Dataset rows: `21`
- Training normal clips: `10`
- Evaluation clips: `11`
- Evaluation windows: `77`
- Precision / recall / F1: `0.792` / `1.000` / `0.884`
- ROC-AUC: `0.990`

## Confusion Matrix

| TP | TN | FP | FN |
|---:|---:|---:|---:|
| 42 | 24 | 11 | 0 |

## Notes

This report uses the same window feature contract as the streaming service.
For MIMII/ToyADMOS-style data, train on normal clips and evaluate on both
normal and abnormal clips. The included generated sample validates the
adapter offline; real downloaded data should be used for final claims.

## Recent Scored Windows

| File | Window | Label | Score | Threshold | Anomaly |
|---|---:|---|---:|---:|---|
| abnormal_test_00.wav | 0 | anomaly | 75.993 | 8.814 | True |
| abnormal_test_00.wav | 1 | anomaly | 81.700 | 8.814 | True |
| abnormal_test_00.wav | 2 | anomaly | 80.811 | 8.814 | True |
| abnormal_test_00.wav | 3 | anomaly | 80.012 | 8.814 | True |
| abnormal_test_00.wav | 4 | anomaly | 79.208 | 8.814 | True |
| abnormal_test_00.wav | 5 | anomaly | 80.246 | 8.814 | True |
| abnormal_test_00.wav | 6 | anomaly | 80.140 | 8.814 | True |
| abnormal_test_01.wav | 0 | anomaly | 16.929 | 8.814 | True |
| abnormal_test_01.wav | 1 | anomaly | 16.227 | 8.814 | True |
| abnormal_test_01.wav | 2 | anomaly | 16.961 | 8.814 | True |
| abnormal_test_01.wav | 3 | anomaly | 18.069 | 8.814 | True |
| abnormal_test_01.wav | 4 | anomaly | 17.181 | 8.814 | True |
| abnormal_test_01.wav | 5 | anomaly | 18.124 | 8.814 | True |
| abnormal_test_01.wav | 6 | anomaly | 17.524 | 8.814 | True |
| abnormal_test_02.wav | 0 | anomaly | 41.242 | 8.814 | True |
| abnormal_test_02.wav | 1 | anomaly | 40.326 | 8.814 | True |
| abnormal_test_02.wav | 2 | anomaly | 38.793 | 8.814 | True |
| abnormal_test_02.wav | 3 | anomaly | 40.835 | 8.814 | True |
| abnormal_test_02.wav | 4 | anomaly | 41.962 | 8.814 | True |
| abnormal_test_02.wav | 5 | anomaly | 40.275 | 8.814 | True |
| abnormal_test_02.wav | 6 | anomaly | 39.085 | 8.814 | True |
| abnormal_test_03.wav | 0 | anomaly | 24.076 | 8.814 | True |
| abnormal_test_03.wav | 1 | anomaly | 25.518 | 8.814 | True |
| abnormal_test_03.wav | 2 | anomaly | 25.163 | 8.814 | True |
| abnormal_test_03.wav | 3 | anomaly | 22.608 | 8.814 | True |
| abnormal_test_03.wav | 4 | anomaly | 27.373 | 8.814 | True |
| abnormal_test_03.wav | 5 | anomaly | 26.777 | 8.814 | True |
| abnormal_test_03.wav | 6 | anomaly | 24.735 | 8.814 | True |
| abnormal_test_04.wav | 0 | anomaly | 30.544 | 8.814 | True |
| abnormal_test_04.wav | 1 | anomaly | 29.424 | 8.814 | True |
| abnormal_test_04.wav | 2 | anomaly | 28.610 | 8.814 | True |
| abnormal_test_04.wav | 3 | anomaly | 27.718 | 8.814 | True |
| abnormal_test_04.wav | 4 | anomaly | 27.526 | 8.814 | True |
| abnormal_test_04.wav | 5 | anomaly | 26.915 | 8.814 | True |
| abnormal_test_04.wav | 6 | anomaly | 27.446 | 8.814 | True |
| abnormal_test_05.wav | 0 | anomaly | 28.193 | 8.814 | True |
| abnormal_test_05.wav | 1 | anomaly | 30.945 | 8.814 | True |
| abnormal_test_05.wav | 2 | anomaly | 30.158 | 8.814 | True |
| abnormal_test_05.wav | 3 | anomaly | 28.592 | 8.814 | True |
| abnormal_test_05.wav | 4 | anomaly | 26.768 | 8.814 | True |
| abnormal_test_05.wav | 5 | anomaly | 27.613 | 8.814 | True |
| abnormal_test_05.wav | 6 | anomaly | 28.575 | 8.814 | True |
| normal_test_00.wav | 0 | normal | 20.423 | 8.814 | True |
| normal_test_00.wav | 1 | normal | 20.413 | 8.814 | True |
| normal_test_00.wav | 2 | normal | 4.417 | 8.814 | False |
| normal_test_00.wav | 3 | normal | 4.745 | 8.814 | False |
| normal_test_00.wav | 4 | normal | 4.953 | 8.814 | False |
| normal_test_00.wav | 5 | normal | 5.098 | 8.814 | False |
| normal_test_00.wav | 6 | normal | 4.966 | 8.814 | False |
| normal_test_01.wav | 0 | normal | 12.343 | 8.814 | True |
| normal_test_01.wav | 1 | normal | 12.723 | 8.814 | True |
| normal_test_01.wav | 2 | normal | 12.079 | 8.814 | True |
| normal_test_01.wav | 3 | normal | 12.184 | 8.814 | True |
| normal_test_01.wav | 4 | normal | 10.867 | 8.814 | True |
| normal_test_01.wav | 5 | normal | 11.685 | 8.814 | True |
| normal_test_01.wav | 6 | normal | 12.674 | 8.814 | True |
| normal_test_02.wav | 0 | normal | 6.677 | 8.814 | False |
| normal_test_02.wav | 1 | normal | 6.774 | 8.814 | False |
| normal_test_02.wav | 2 | normal | 4.407 | 8.814 | False |
| normal_test_02.wav | 3 | normal | 5.020 | 8.814 | False |
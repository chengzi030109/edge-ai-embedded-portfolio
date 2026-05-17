# MCU Resource Budget

This report estimates the resource footprint of the centroid detector
when moved from the laptop prototype into MCU firmware.

## Model And Buffer Sizes

| Item | Bytes | Notes |
|---|---:|---|
| Raw sample window (`int16_t`) | 512 | 256 samples |
| Feature vector (`float`) | 40 | Current float C path |
| Feature vector (`Q24.8 int32_t`) | 40 | Fixed-point inference path |
| Float centroid parameters | 84 | mean + scale + threshold |
| Q24.8 centroid parameters | 84 | mean_q + inv_scale_q + threshold_q |
| Compact telemetry estimate | 20 | Binary MCU-to-gateway frame estimate |

## Fixed-Point Validation

- Q format: `Q24.8`
- threshold_q: `1559`
- parameter-only decision mismatches: `0`
- integer-path decision mismatches: `0`
- parameter-only mean absolute score error: `0.1406357169151306`
- integer-path mean absolute score error: `0.30370343178510667`

## Per-Window Inference Work

For each feature, fixed-point centroid inference performs one subtract,
one normalization multiply, one shift, one square multiply, and one
accumulate. Each window then performs one integer square root and one
threshold comparison.

Feature extraction, especially FFT band power, is expected to dominate
runtime. The centroid scoring stage is intentionally small enough to fit
comfortably inside a periodic RTOS inference task.

## Interview Reading

The important engineering point is that the project now has both a float C
path for clarity and a Q-format C path for MCU realism. The next true
firmware step is to make feature extraction produce Q-format values
directly, then replace the laptop replay source with a sensor driver.

# Fixed-Point Centroid Report

- Vectors evaluated: `80`
- Float parameter footprint: `84` bytes
- Fixed-point parameter footprint: `84` bytes
- Decision mismatches: `0`
- Mean absolute score error: `1.406357e-01`
- Max absolute score error: `3.034668e-01`

## Quantization Parameters

| Parameter | Value |
|---|---:|
| Q fractional bits | 8 |
| Q step | 3.906250e-03 |
| threshold_q | 1559 |

This report simulates storing centroid parameters in Q24.8 int32 format.
The current firmware path still uses float inference; fixed-point
is the next MCU optimization step after functional parity.

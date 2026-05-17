# TinyML Predictive Maintenance Portfolio Summary

This is the short, interview-facing index for the project. It connects
the laptop prototype, the repeatable demo commands, and the MCU/TinyML
migration story in one place.

## Result Snapshot

| Area | What to Look At | Current Result |
|---|---|---:|
| Synthetic simulator | Normal-only anomaly detection | F1 `0.9831`, accuracy `0.9750` |
| Fixed-point simulation | Q-format centroid drift | `0` parameter mismatches, `0` integer-path mismatches |
| PHM2008/C-MAPSS sample | Harder gradual degradation task | F1 `0.9600`, FPR `0.2727` |
| CWRU bearing data | Real-data sanity check, ceiling warning | F1 `1.0000` |
| C inference parity | Firmware-facing float centroid path | covered by `tests/test_c_inference_parity.py` |
| Fixed C inference parity | Q24.8 MCU-style path | covered by `tests/test_c_fixed_inference_parity.py` |

## Demo Command

```powershell
.\.venv\Scripts\python.exe scripts\run_portfolio_demo.py --quick
```

The demo trains a fresh synthetic model, runs both synthetic and CSV replay
telemetry, evaluates the model, refreshes fixed-point and PHM2008 reports,
regenerates figures, and rewrites this summary.

## Figures

![Portfolio pipeline](figures/portfolio_pipeline.png)

![Synthetic score curve](figures/synthetic_score_curve.png)

![Alarm debounce timeline](figures/alarm_debounce_timeline.png)

![CWRU model comparison](figures/cwru_model_comparison.png)

![Quantization size and latency](figures/quantization_size_latency.png)

![PHM2008 comparison](figures/phm2008_comparison.png)

## Engineering Read

- The synthetic result proves the whole edge pipeline is wired correctly:
  simulator/replay, windowing, FFT/statistical features, anomaly score,
  debounce, telemetry, and reports.
- CWRU is useful but close to a ceiling because seeded bearing faults are
  strongly separated in RMS, kurtosis, and spectral-band features. In an
  interview, treat CWRU as a validation dataset, not as the only claim.
- PHM2008/C-MAPSS is the better difficulty story: multivariate engine
  cycles, gradual degradation, and noisier false-positive behavior.
- Fixed-point Q24.8 is now measured in two ways: parameter-only drift
  and an integer C-shaped path with a matching C parity test.

## Deployment Notes

- Quantization formats generated: `4`
- MCU resource report: `84` bytes of Q24.8 parameters.
- MCU boundary: replace simulator/CSV replay with sensor drivers, keep the
  feature vector and detector contract stable.
- Next firmware depth: CMSIS-DSP FFT, direct Q-format feature extraction,
  FreeRTOS task split, and UART/MQTT telemetry on an embedded-Linux gateway.

## Resume Bullets

- Built a hardware-free TinyML predictive-maintenance demo with RTOS-style
  sensor, feature, inference, alarm, and telemetry stages.
- Implemented CSV replay, alarm debounce, fixed-point drift analysis, C
  inference parity tests, Q24.8 integer inference, and MCU resource reports.
- Evaluated the same embedded-friendly anomaly detector on synthetic
  vibration, CWRU bearing data, and PHM2008/C-MAPSS-style degradation
  windows, with honest discussion of dataset difficulty.

## Follow-Up Reports

- [`evaluation.md`](evaluation.md)
- [`fixed_point_report.md`](fixed_point_report.md)
- [`phm2008_comparison.md`](phm2008_comparison.md)
- [`cwru_comparison.md`](cwru_comparison.md)
- [`quantization_report.md`](quantization_report.md)
- [`mcu_resource_budget.md`](mcu_resource_budget.md)
- [`../docs/mcu-migration.md`](../docs/mcu-migration.md)
- [`../docs/interview-notes.md`](../docs/interview-notes.md)
- [`../docs/resume-bullets.md`](../docs/resume-bullets.md)

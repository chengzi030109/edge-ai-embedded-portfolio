# Firmware Migration Prototype

This folder contains small C prototypes for future MCU migration.

The Python implementation is the source of truth for the laptop simulation, but
the C files show how the feature stage can move to:

- ESP32-S3 + FreeRTOS
- STM32 + FreeRTOS
- Zephyr on Cortex-M
- bare-metal firmware

## Files

- `feature_extract.h`: portable C API
- `feature_extract.c`: no-allocation time-domain feature implementation
- `inference.h`: portable C centroid inference API
- `inference.c`: float centroid scoring and thresholding
- `model_params.h`: generated model constants from `scripts/export_model_to_c.py`
- `test_inference.c`: host-side parity harness used by tests/CI

## Why Only Time-Domain Features?

FFT support depends on the target platform and DSP library, such as CMSIS-DSP on
Arm Cortex-M. Time-domain features are useful first because they are cheap, easy
to test, and map directly to the Python prototype:

- mean
- RMS
- standard deviation
- peak-to-peak
- crest factor

Later, FFT band-power features can be added with CMSIS-DSP or another embedded
DSP library.

## Float First, Fixed-Point Next

The current C inference path intentionally uses float arithmetic because it is
the clearest way to prove Python/C numerical parity. The project also includes
`scripts/fixed_point_report.py`, which simulates int16 centroid parameters and
reports score drift and decision mismatches. That report is the decision point
before writing a true fixed-point C implementation.

# Firmware Migration Prototype

This folder contains a small C feature-extraction prototype for future MCU
migration.

The Python implementation is the source of truth for the laptop simulation, but
the C files show how the feature stage can move to:

- ESP32-S3 + FreeRTOS
- STM32 + FreeRTOS
- Zephyr on Cortex-M
- bare-metal firmware

## Files

- `feature_extract.h`: portable C API
- `feature_extract.c`: no-allocation time-domain feature implementation

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


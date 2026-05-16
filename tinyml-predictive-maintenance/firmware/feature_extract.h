#ifndef TPM_FEATURE_EXTRACT_H
#define TPM_FEATURE_EXTRACT_H

/*
 * Minimal C feature extraction interface for MCU migration.
 *
 * The Python project uses FFT features too, but this C prototype starts with
 * time-domain features that are cheap and portable on small MCUs:
 *   - mean
 *   - RMS after DC removal
 *   - standard deviation
 *   - peak-to-peak
 *   - crest factor
 *
 * This file is not tied to a specific RTOS or board. It can be included from a
 * FreeRTOS task, Zephyr thread, bare-metal loop, or unit-test harness.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float mean;
    float rms;
    float stddev;
    float peak_to_peak;
    float crest_factor;
} tpm_time_features_t;

/*
 * Extract time-domain features from one vibration window.
 *
 * samples:
 *   Pointer to float samples. A real firmware port can convert ADC int16 data
 *   to float before calling this function, or create a fixed-point variant.
 *
 * length:
 *   Number of samples in the window. Should match the Python-side window_size
 *   when comparing outputs.
 *
 * out:
 *   Destination struct for extracted features.
 *
 * Returns:
 *   0 on success
 *  -1 if inputs are invalid
 */
int tpm_extract_time_features(const float *samples, size_t length, tpm_time_features_t *out);

#ifdef __cplusplus
}
#endif

#endif /* TPM_FEATURE_EXTRACT_H */

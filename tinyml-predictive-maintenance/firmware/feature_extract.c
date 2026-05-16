#include "feature_extract.h"

/*
 * C prototype for MCU-side feature extraction.
 *
 * This intentionally avoids dynamic allocation, recursion, and platform APIs.
 * That keeps it suitable for small embedded targets and easy to port into an
 * RTOS task. The implementation mirrors the Python time-domain features in
 * src/tpm/features.py.
 */

#include <math.h>

int tpm_extract_time_features(const float *samples, size_t length, tpm_time_features_t *out) {
    if (samples == 0 || out == 0 || length == 0) {
        return -1;
    }

    /*
     * First pass: mean, min, and max. Keeping min/max in the first pass avoids
     * scanning the window again for peak-to-peak.
     */
    float sum = 0.0f;
    float min_value = samples[0];
    float max_value = samples[0];

    for (size_t i = 0; i < length; ++i) {
        float value = samples[i];
        sum += value;
        if (value < min_value) {
            min_value = value;
        }
        if (value > max_value) {
            max_value = value;
        }
    }

    float mean = sum / (float)length;

    /*
     * Second pass: remove DC offset and compute energy. This matches the Python
     * project, where vibration strength is measured around the window mean.
     */
    float centered_energy = 0.0f;
    float abs_peak = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        float centered = samples[i] - mean;
        float abs_centered = fabsf(centered);
        centered_energy += centered * centered;
        if (abs_centered > abs_peak) {
            abs_peak = abs_centered;
        }
    }

    float variance = centered_energy / (float)length;
    float rms = sqrtf(variance);
    float eps = 1.0e-8f;

    out->mean = mean;
    out->rms = rms;
    out->stddev = rms;
    out->peak_to_peak = max_value - min_value;
    out->crest_factor = abs_peak / (rms + eps);
    return 0;
}

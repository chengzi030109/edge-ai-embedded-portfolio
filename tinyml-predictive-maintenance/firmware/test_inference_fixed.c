/*
 * Host-side test harness for the Q24.8 fixed-point inference path.
 *
 * Reads TPM_FIXED_MODEL_N_FEATURES little-endian float32 values from stdin,
 * quantizes them to Q24.8 using the same helper firmware can use during
 * bring-up, runs integer centroid inference, and prints:
 *
 *   <score_q> <is_anomaly>
 *
 * The Python parity test compares this output with src/tpm/fixed_point.py.
 */

#include <stdint.h>
#include <stdio.h>

#include "inference_fixed.h"
#include "model_params_fixed.h"

int main(void) {
    float features[TPM_FIXED_MODEL_N_FEATURES];
    int32_t features_q[TPM_FIXED_MODEL_N_FEATURES];

#ifdef _WIN32
    /* Avoid CRLF translation on Windows so binary float bytes survive intact. */
    extern int _setmode(int, int);
    _setmode(0, 0x8000); /* _O_BINARY = 0x8000 */
#endif

    size_t got = fread(features, sizeof(float), TPM_FIXED_MODEL_N_FEATURES, stdin);
    if (got != TPM_FIXED_MODEL_N_FEATURES) {
        fprintf(stderr,
                "expected %u floats on stdin, got %zu\n",
                (unsigned)TPM_FIXED_MODEL_N_FEATURES,
                got);
        return 2;
    }

    for (size_t i = 0; i < TPM_FIXED_MODEL_N_FEATURES; ++i) {
        features_q[i] = tpm_q24_8_from_float(features[i]);
    }

    tpm_fixed_inference_result_t result;
    int rc = tpm_centroid_predict_fixed(features_q,
                                        TPM_FIXED_MODEL_N_FEATURES,
                                        &g_centroid_fixed_model,
                                        &result);
    if (rc != 0) {
        fprintf(stderr, "tpm_centroid_predict_fixed returned %d\n", rc);
        return 3;
    }

    printf("%d %d\n", result.score_q, result.is_anomaly);
    return 0;
}

/*
 * Host-side test harness for the firmware inference path.
 *
 * Reads a feature vector from stdin (TPM_MODEL_N_FEATURES little-endian
 * float32 values), runs centroid inference, and prints the score and
 * is_anomaly flag to stdout in a parseable format.
 *
 * The Python parity test in tests/test_c_inference_parity.py compiles this
 * file, generates random feature vectors, and compares the C output to the
 * Python detector output. Numerical agreement validates that the firmware
 * port matches the trained model.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "inference.h"
#include "model_params.h"

int main(void) {
    float features[TPM_MODEL_N_FEATURES];

    /*
     * Use binary stdin so we do not have to parse decimal floats. fread also
     * sidesteps locale issues on Windows hosts where scanf("%f") may interpret
     * "1.5" differently depending on LC_NUMERIC.
     */
#ifdef _WIN32
    /* Avoid CRLF translation on Windows so binary float bytes survive intact. */
    extern int _setmode(int, int);
    _setmode(0, 0x8000); /* _O_BINARY = 0x8000 */
#endif

    size_t got = fread(features, sizeof(float), TPM_MODEL_N_FEATURES, stdin);
    if (got != TPM_MODEL_N_FEATURES) {
        fprintf(stderr,
                "expected %u floats on stdin, got %zu\n",
                (unsigned)TPM_MODEL_N_FEATURES,
                got);
        return 2;
    }

    tpm_inference_result_t result;
    int rc = tpm_centroid_predict(features,
                                  TPM_MODEL_N_FEATURES,
                                  &g_centroid_model,
                                  &result);
    if (rc != 0) {
        fprintf(stderr, "tpm_centroid_predict returned %d\n", rc);
        return 3;
    }

    /*
     * Print "<score> <is_anomaly>" so the Python side can split on whitespace.
     * Use plenty of digits so float32 round-trip differences are visible if
     * they exist.
     */
    printf("%.9g %d\n", result.score, result.is_anomaly);
    return 0;
}

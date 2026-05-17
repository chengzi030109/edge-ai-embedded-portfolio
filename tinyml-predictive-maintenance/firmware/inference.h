#ifndef TPM_INFERENCE_H
#define TPM_INFERENCE_H

/*
 * Centroid anomaly inference for MCU deployment.
 *
 * The Python project ships the same logic in src/tpm/model.py. This C version
 * is a direct port: per-feature normalization to z-score, L2 distance, and
 * threshold comparison.
 *
 * Model parameters live in a tpm_centroid_model_t struct, typically generated
 * by scripts/export_model_to_c.py and included as a const header. That keeps
 * the implementation free of dynamic allocation and JSON parsing, which is
 * the right choice on a small MCU.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Number of features the model expects. Used to bounds-check callers. */
    size_t n_features;

    /* Per-feature mean and scale arrays. Both are length n_features. */
    const float *mean;
    const float *scale;

    /* Threshold above which a window is flagged as anomalous. */
    float threshold;
} tpm_centroid_model_t;

typedef struct {
    float score;
    int   is_anomaly;
} tpm_inference_result_t;

/*
 * Run centroid anomaly inference on one feature vector.
 *
 * features:  Pointer to the feature vector. Length must equal model->n_features.
 * length:    Length of the features array. Validated against the model.
 * model:     Pointer to a populated centroid model.
 * out:       Destination for score and is_anomaly. Must not be NULL.
 *
 * Returns 0 on success, -1 on invalid input.
 */
int tpm_centroid_predict(const float *features,
                         size_t length,
                         const tpm_centroid_model_t *model,
                         tpm_inference_result_t *out);

#ifdef __cplusplus
}
#endif

#endif /* TPM_INFERENCE_H */

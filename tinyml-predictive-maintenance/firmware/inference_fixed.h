#ifndef TPM_INFERENCE_FIXED_H
#define TPM_INFERENCE_FIXED_H

/*
 * Q24.8 fixed-point centroid inference for MCU deployment.
 *
 * This file is the production-shaped companion to inference.c. The float path
 * is easier to debug; this fixed path is closer to what a small MCU without a
 * fast FPU would run. All feature values and model parameters use signed Q24.8:
 *
 *   real_value = q_value / 256.0
 *
 * The model stores reciprocal scale instead of scale so inference uses
 * multiply + shift instead of division:
 *
 *   z_q = ((feature_q - mean_q) * inv_scale_q) >> 8
 *
 * z_q remains Q24.8. Squaring and summing z_q values gives a Q48.16 distance
 * squared. Integer sqrt returns a Q24.8 score, directly comparable with the
 * Q24.8 threshold.
 */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TPM_Q24_8_FRACTIONAL_BITS 8
#define TPM_Q24_8_SCALE           (1 << TPM_Q24_8_FRACTIONAL_BITS)

typedef struct {
    /* Number of features the model expects. Used to bounds-check callers. */
    size_t n_features;

    /* Per-feature Q24.8 mean and reciprocal-scale arrays. */
    const int32_t *mean_q;
    const int32_t *inv_scale_q;

    /* Q24.8 anomaly threshold. */
    int32_t threshold_q;
} tpm_centroid_fixed_model_t;

typedef struct {
    /* Q24.8 anomaly score. Divide by 256.0 for a human-readable float. */
    int32_t score_q;

    /* 1 when score_q > threshold_q, otherwise 0. */
    int is_anomaly;
} tpm_fixed_inference_result_t;

/*
 * Convert a float to Q24.8 using round-to-nearest.
 *
 * This helper is mainly for host-side tests and board bring-up. In a final
 * firmware pipeline, feature extraction should preferably produce Q-format
 * values directly and avoid float conversion entirely.
 */
int32_t tpm_q24_8_from_float(float value);

/* Convert Q24.8 to float for logs, tests, and debugging. */
float tpm_q24_8_to_float(int32_t value_q);

/*
 * Run fixed-point centroid anomaly inference on one quantized feature vector.
 *
 * features_q: Q24.8 feature vector. Length must equal model->n_features.
 * length:     Length of features_q. Validated against the model.
 * model:      Q24.8 centroid model.
 * out:        Destination for Q24.8 score and anomaly flag.
 *
 * Returns 0 on success, -1 on invalid input.
 */
int tpm_centroid_predict_fixed(const int32_t *features_q,
                               size_t length,
                               const tpm_centroid_fixed_model_t *model,
                               tpm_fixed_inference_result_t *out);

#ifdef __cplusplus
}
#endif

#endif /* TPM_INFERENCE_FIXED_H */

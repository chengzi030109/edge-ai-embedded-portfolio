#include "inference.h"

/*
 * Centroid anomaly inference, ported from src/tpm/model.py.
 *
 * No dynamic allocation, no platform APIs. Suitable for a FreeRTOS task,
 * Zephyr thread, or bare-metal main loop. The implementation is intentionally
 * straightforward so it can be reasoned about line-for-line with the Python
 * version when validating numerical parity.
 */

#include <math.h>
#include <stddef.h>

int tpm_centroid_predict(const float *features,
                         size_t length,
                         const tpm_centroid_model_t *model,
                         tpm_inference_result_t *out) {
    if (features == 0 || model == 0 || out == 0) {
        return -1;
    }
    if (length != model->n_features) {
        return -1;
    }
    if (model->mean == 0 || model->scale == 0) {
        return -1;
    }

    /*
     * z = (x - mean) / scale
     * score = sqrt(sum(z * z))
     *
     * The Python side adds a tiny epsilon to scale before training so the
     * stored values are already safe to divide by. We do not add another
     * epsilon here; doing so would diverge from the trained threshold.
     */
    float sum_squares = 0.0f;
    for (size_t i = 0; i < length; ++i) {
        float z = (features[i] - model->mean[i]) / model->scale[i];
        sum_squares += z * z;
    }
    float score = sqrtf(sum_squares);

    out->score = score;
    out->is_anomaly = (score > model->threshold) ? 1 : 0;
    return 0;
}

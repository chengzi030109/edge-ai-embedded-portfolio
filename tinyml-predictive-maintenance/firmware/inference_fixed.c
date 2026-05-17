#include "inference_fixed.h"

/*
 * Q24.8 fixed-point centroid anomaly inference.
 *
 * The implementation intentionally avoids malloc, printf, and platform APIs so
 * it can be lifted into a FreeRTOS task or bare-metal loop. The only helper
 * that touches float is tpm_q24_8_from_float(), which exists for host-side
 * parity tests and first-board bring-up. The actual inference function consumes
 * already-quantized int32_t features.
 */

#include <limits.h>
#include <stdint.h>

static uint64_t tpm_isqrt_u64(uint64_t value) {
    /*
     * Binary restoring square root. It returns floor(sqrt(value)) using only
     * shifts, comparisons, and subtraction. That makes the score deterministic
     * across C libraries and MCU toolchains.
     */
    uint64_t root = 0;
    uint64_t bit = (uint64_t)1 << 62;

    while (bit > value) {
        bit >>= 2;
    }

    while (bit != 0) {
        if (value >= root + bit) {
            value -= root + bit;
            root = (root >> 1) + bit;
        } else {
            root >>= 1;
        }
        bit >>= 2;
    }

    return root;
}

int32_t tpm_q24_8_from_float(float value) {
    /*
     * Round to nearest without depending on libm's lrintf/roundf. Positive and
     * negative values need opposite half-step signs to round away from zero at
     * .5, matching the practical behavior expected by the Python export tests.
     */
    double scaled = (double)value * (double)TPM_Q24_8_SCALE;
    if (scaled >= (double)INT32_MAX) {
        return INT32_MAX;
    }
    if (scaled <= (double)INT32_MIN) {
        return INT32_MIN;
    }
    if (scaled >= 0.0) {
        return (int32_t)(scaled + 0.5f);
    }
    return (int32_t)(scaled - 0.5f);
}

float tpm_q24_8_to_float(int32_t value_q) {
    return (float)value_q / (float)TPM_Q24_8_SCALE;
}

int tpm_centroid_predict_fixed(const int32_t *features_q,
                               size_t length,
                               const tpm_centroid_fixed_model_t *model,
                               tpm_fixed_inference_result_t *out) {
    if (features_q == 0 || model == 0 || out == 0) {
        return -1;
    }
    if (length != model->n_features) {
        return -1;
    }
    if (model->mean_q == 0 || model->inv_scale_q == 0) {
        return -1;
    }

    uint64_t sum_squares_q16 = 0;
    for (size_t i = 0; i < length; ++i) {
        /*
         * diff_q and inv_scale_q are both Q24.8. Their product is Q48.16; the
         * right shift brings z back to Q24.8. The expected feature ranges are
         * small enough that int64_t is ample for the intermediate product.
         */
        int64_t diff_q = (int64_t)features_q[i] - (int64_t)model->mean_q[i];
        int64_t z_q = (diff_q * (int64_t)model->inv_scale_q[i]) >> TPM_Q24_8_FRACTIONAL_BITS;

        /*
         * z_q * z_q is Q48.16. We accumulate in uint64_t because the value is a
         * squared distance and therefore non-negative.
         */
        sum_squares_q16 += (uint64_t)(z_q * z_q);
    }

    uint64_t score_q = tpm_isqrt_u64(sum_squares_q16);
    if (score_q > (uint64_t)INT32_MAX) {
        score_q = (uint64_t)INT32_MAX;
    }

    out->score_q = (int32_t)score_q;
    out->is_anomaly = (out->score_q > model->threshold_q) ? 1 : 0;
    return 0;
}

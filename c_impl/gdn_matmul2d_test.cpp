/*
 * gdn_matmul2d_test.cpp — verify gdn_matmul2d_top (the standard tiled
 * output-stationary systolic matmul) against a naive native-C matmul.
 *
 * Identical harness to gdn_matmul_test.cpp, but drives gdn_matmul2d_top so
 * the two kernels can be parity-checked and (via test_matmul2d.tcl)
 * synthesized head-to-head.
 *
 * Usage:
 *   ./gdn_matmul2d_test [num_rows] [in_dim] [out_dim] [seed]
 *
 * Exit 0 on PASS, 1 on FAIL.
 */

#include "gdn_model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_NUM_ROWS  2048
#define DEFAULT_IN_DIM    2048
#define DEFAULT_OUT_DIM   2048
#define DEFAULT_SEED      42u

#define ATOL 1e-3f
#define RTOL 1e-3f

static float rand_float(void) {
    return ((float)rand() / (float)RAND_MAX) * 2.0f - 1.0f;
}

/* out[r, c] = sum_k in[r, k] * weights[c, k]  (weights row-major out_dim x in_dim) */
static void naive_matmul(
    float *out,
    const float *in,
    const float *weights,
    uint32_t num_rows,
    uint32_t in_dim,
    uint32_t out_dim
) {
    for (uint32_t r = 0; r < num_rows; ++r) {
        for (uint32_t c = 0; c < out_dim; ++c) {
            float sum = 0.0f;
            for (uint32_t k = 0; k < in_dim; ++k) {
                sum += in[(size_t)r * in_dim + k] *
                       weights[(size_t)c * in_dim + k];
            }
            out[(size_t)r * out_dim + c] = sum;
        }
    }
}

int main(int argc, char **argv) {
    uint32_t num_rows = DEFAULT_NUM_ROWS;
    uint32_t in_dim   = DEFAULT_IN_DIM;
    uint32_t out_dim  = DEFAULT_OUT_DIM;
    uint32_t seed     = DEFAULT_SEED;

    if (argc > 1) num_rows = (uint32_t)strtoul(argv[1], NULL, 10);
    if (argc > 2) in_dim   = (uint32_t)strtoul(argv[2], NULL, 10);
    if (argc > 3) out_dim  = (uint32_t)strtoul(argv[3], NULL, 10);
    if (argc > 4) seed     = (uint32_t)strtoul(argv[4], NULL, 10);

    if (num_rows == 0 || in_dim == 0 || out_dim == 0) {
        fprintf(stderr, "FATAL: all dimensions must be > 0\n");
        return 1;
    }

    fprintf(stderr, "matmul2d test: %u x %u x %u  (seed=%u)\n",
            num_rows, in_dim, out_dim, seed);

    size_t in_count  = (size_t)num_rows * in_dim;
    size_t wt_count  = (size_t)out_dim  * in_dim;
    size_t out_count = (size_t)num_rows * out_dim;

    float *in_buf     = (float *)malloc(in_count  * sizeof(float));
    float *wt_buf     = (float *)malloc(wt_count  * sizeof(float));
    float *out_hls    = (float *)malloc(out_count * sizeof(float));
    float *out_golden = (float *)malloc(out_count * sizeof(float));
    if (!in_buf || !wt_buf || !out_hls || !out_golden) {
        fprintf(stderr, "FATAL: allocation failed\n");
        free(in_buf); free(wt_buf); free(out_hls); free(out_golden);
        return 1;
    }

    size_t i;
    srand(seed);
    for (i = 0; i < in_count; ++i) in_buf[i] = rand_float();
    for (i = 0; i < wt_count; ++i) wt_buf[i] = rand_float();

    fprintf(stderr, "computing golden (naive matmul)...\n");
    naive_matmul(out_golden, in_buf, wt_buf, num_rows, in_dim, out_dim);

    fprintf(stderr, "calling gdn_matmul2d_top...\n");
    int rc = gdn_matmul2d_top(out_hls, in_buf, wt_buf,
                              num_rows, in_dim, out_dim);
    if (rc != 0) {
        fprintf(stderr, "FATAL: gdn_matmul2d_top returned %d\n", rc);
        free(in_buf); free(wt_buf); free(out_hls); free(out_golden);
        return 1;
    }

    fprintf(stderr, "comparing %zu output values...\n", out_count);
    float    max_abs_diff = 0.0f;
    float    max_rel_diff = 0.0f;
    uint32_t fail_count   = 0;
    for (i = 0; i < out_count; ++i) {
        float diff  = fabsf(out_hls[i] - out_golden[i]);
        float denom = fmaxf(fabsf(out_golden[i]), 1e-8f);
        float rel   = diff / denom;

        if (diff > max_abs_diff) max_abs_diff = diff;
        if (rel  > max_rel_diff) max_rel_diff = rel;

        if (diff > ATOL + RTOL * fabsf(out_golden[i])) {
            if (fail_count < 10) {
                uint32_t r = (uint32_t)(i / out_dim);
                uint32_t c = (uint32_t)(i % out_dim);
                fprintf(stderr,
                        "  MISMATCH [r=%u c=%u]: hls=%.8f golden=%.8f diff=%.2e\n",
                        r, c, out_hls[i], out_golden[i], diff);
            }
            ++fail_count;
        }
    }

    printf("--- gdn_matmul2d Parity Test ---\n");
    printf("dims:         %u x %u x %u\n", num_rows, in_dim, out_dim);
    printf("seed:         %u\n", seed);
    printf("total_values: %zu\n", out_count);
    printf("max_abs_diff: %.6e\n", max_abs_diff);
    printf("max_rel_diff: %.6e\n", max_rel_diff);
    printf("fail_count:   %u  (atol=%.0e rtol=%.0e)\n", fail_count, ATOL, RTOL);
    printf("RESULT: %s\n", fail_count == 0 ? "PASS" : "FAIL");

    free(in_buf);
    free(wt_buf);
    free(out_hls);
    free(out_golden);
    return fail_count == 0 ? 0 : 1;
}

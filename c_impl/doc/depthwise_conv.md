# Depthwise 1D Convolution + SiLU

**Status:** Iter39C production decode block. Synthesis tables in the lower
historical section are from the retired prefill top.

**Location:** `gdn_model.cpp` (`gdn_depthwise_conv_silu_head_kind`)

## Overview

Causal depthwise 1D convolution applied independently to each of the 2048
channels of Q, K, V (after their linear projections), followed by a SiLU
activation:

```
out[r][c] = SiLU( Σ_k in[r - 3 + k][c] * weights[c][k] )    for k in 0..3
```

where rows before the current token come from the persistent convolution tail.
For the first decode token that tail is imported from the GPU prefill state;
after every call the newest three input rows are written back for the next
token.

Iter39C no longer calls a whole-hidden function three times after QKVG. The
QKVG collector emits one complete 256-element head every two 32-channel result
rounds. A bounded sink then invokes one time-shared 256-column actor for Q, K,
and V before consuming the next head. The actor's convolution runs while later
heads are still being produced by the upstream GEMV dataflow graph.

## Current Iter39C Data Movement

Before QKVG, six fixed-bank loops stage the three 32 KiB convolution-weight
tensors and three 24 KiB old-tail tensors from HBM0 into partitioned BRAM. HLS
infers 512-bit bursts and II=1 for every loop; the combined context load is
3,137 cycles/layer.

For each head, the actor:

1. loads that head's 64 packed weight words;
2. restores 48 packed old-tail words;
3. shifts in 16 packed words from the raw Q, K, or V result;
4. computes 256 four-tap convolutions with four element lanes; and
5. stores the convolved head into its local activation buffer.

After all three kinds consume a head, the sink reuses obsolete tail row 0 for
the new raw Q/K/V row. Three final fixed-bank stores emit old rows 1/2 followed
by that new row in 1,371 cycles/layer. This preserves the `.gdnstate` tail ABI,
FP32 order, and exact multi-token behavior without allocating another tail
buffer.

Integrated Iter39C synthesis reports one physical head-convolution actor,
512-bit context movers at II=1, unchanged GEMV MM2S II=1/MAC II=4, and a
473,688-cycle fixed reduction versus Iter38. The routed 100 MHz image measures
4.309M cycles/token versus 4.708M for Iter38, with exact 64-token parity.

## Historical Whole-Tensor Implementation

The remaining sections document the earlier generic whole-tensor helper and
the prefill optimization history; they are not the current decode schedule.

## Compile-time bounds

```c
#define GDN_CONV_COLS_MAX 2048   // hidden, always 2048 for GDN-1.3B
#define GDN_CONV_K_MAX    4      // conv_size, always 4
```

These bound the local buffers; the runtime `num_cols` and `kernel_size` may be
smaller, but never larger.

## Optimisation 1: Pre-Buffered Weights

The original loop body read the per-tap weight on every iteration:

```c
sum += in[..][col] * weights[col * kernel_size + k];   // m_axi load!
```

That issued a fresh `m_axi` read of `weights[]` every tap, every column, every
row — 4 × 2048 × 2048 = 16 M reads per call, even though there are only 8192
distinct weight values. The current implementation pulls them on-chip once at
the top of the function:

```c
float w_loc[GDN_CONV_COLS_MAX][GDN_CONV_K_MAX];
#pragma HLS array_partition variable=w_loc dim=2 complete

conv_load_w_col: for (col = 0; col < num_cols; ++col) {
    conv_load_w_k: for (k = 0; k < kernel_size; ++k) {
    #pragma HLS pipeline II=1
        w_loc[col][k] = weights[(size_t)col * kernel_size + k];
    }
}
```

Cost: 8192 × 4 = 32 KB on chip (≈ 16 BRAM_18K). With `dim 2 complete`
partitioning, the four taps for any `col` can be read in parallel — the inner
compute pipeline gets all four kernel weights in a single cycle.

Latency: 8266 cycles total to fill the buffer, paid once per call.

## Optimisation 2: 4-Row Sliding Window

The activation tensor is too large to buffer fully (`num_rows × num_cols` =
2048 × 2048 = 16 MB), but the kernel only ever needs the last 4 rows. A
sliding window holds those 4 rows on chip:

```c
float in_window[GDN_CONV_K_MAX][GDN_CONV_COLS_MAX];
#pragma HLS array_partition variable=in_window dim=1 complete
```

Cost: 4 × 2048 = 32 KB (≈ 16 BRAM_18K). With `dim 1 complete` partitioning,
each row of the window is in a distinct bank — the four taps for any `col` can
be read in parallel.

The window is **zero-initialised** at the top of the function so the first
`kernel_size − 1` rows naturally produce zero contributions for the
out-of-bounds taps:

```c
conv_init_win_k: for (k = 0; k < kernel_size; ++k) {
#pragma HLS unroll
    conv_init_win_col: for (col = 0; col < num_cols; ++col) {
    #pragma HLS pipeline II=1
        in_window[k][col] = 0.0f;
    }
}
```

This eliminates the boundary `if (source_row >= 0)` test inside the hot loop.

## Optimisation 3: Two-Phase per Row (avoid AXI port hazard)

A naive fused pipeline:

```c
conv_col: for (col ...) {
    #pragma HLS pipeline II=1
    /* shift window, load in[r][col] from m_axi, compute, write out[r][col] to m_axi */
}
```

forces HLS to schedule a `gmem` read and a `gmem` write in the same iteration.
Even though the AXI4 master has independent AR/AW channels, HLS treats writes
followed by reads on the same `gmem` bundle as a carried dep:

```
HLS 200-880  Unable to enforce a carried dependence constraint (II=1, dist=1,
             offset=1) between bus response on gmem and bus request on gmem
```

This produced II=155 (effectively serialised) in iteration v2.

The current implementation splits the per-row work into two passes:

```c
conv_row: for (row = 0; row < num_rows; ++row) {

    /* Phase A: m_axi READ only — pull row r and shift the window. */
    conv_load: for (col = 0; col < num_cols; ++col) {
    #pragma HLS pipeline II=1
        in_window[0][col] = in_window[1][col];
        in_window[1][col] = in_window[2][col];
        in_window[2][col] = in_window[3][col];
        in_window[3][col] = in[(size_t)row * num_cols + col];
    }

    /* Phase B: m_axi WRITE only — MAC against w_loc and emit out[]. */
    conv_compute: for (col = 0; col < num_cols; ++col) {
    #pragma HLS pipeline II=1
        float sum = in_window[0][col] * w_loc[col][0]
                  + in_window[1][col] * w_loc[col][1]
                  + in_window[2][col] * w_loc[col][2]
                  + in_window[3][col] * w_loc[col][3];
        out[(size_t)row * num_cols + col] = gdn_silu(sum);
    }
}
```

Each phase touches only one direction of the `gmem` port, so HLS pipelines both
at II=1.

The 4-tap MAC is a balanced expression `(a + b) + (c + d)` — HLS schedules
this as two parallel adds followed by one final add (depth ≈ 2 fadd stages),
not as a 4-deep serial accumulator chain.

## Historical Synthesis Results (single-layer, U55C @ 100 MHz)

Per-call latency, from `solution2/syn/report/csynth.rpt`:

| Pass | conv_row_conv_col | per-call total | per-call ns |
|------|------------------:|---------------:|------------:|
| v1 (original)  | 759 M (II=N/A, no pipeline) | 759.2 M | 7.59 s |
| v2 (fused)     | 1.74 G (II=155)             | 1.74 G | 17.45 s |
| v3+ (2-phase)  | 8.74 M (conv_load II=1, conv_compute II=1) | **8.75 M** | **87.5 ms** |

The 2-phase split is **86× faster** than v1 and **200× faster** than v2.

Total per call:
- weights pre-load: 8265 cyc (paid once)
- window init: 8194 cyc (paid once)
- main: `num_rows × (conv_load + conv_compute)` = 2048 × (2051 + 2144) ≈ 8.59 M (matmul_1 instance) or 2048 × (2121 + 2144) ≈ 8.74 M (matmul_2 instance)
- **total ≈ 8.75 M cycles ≈ 87.5 ms @ 100 MHz**

For the three calls (Q, K, V) per attention layer, total conv cost is ~26.2 M
cycles — <0.02 % of `gdn_attn_forward`.

## Historical Resource Cost (per instance, U55C)

| Resource | conv_silu_1 (mem_q out) | conv_silu_2 (mem_k / gmem out) |
|----------|------------------------:|--------------------------------:|
| BRAM_18K | 32                      | 32                              |
| DSP      | 36                      | 6                               |
| FF       | 4.9 k                   | 4.1 k                           |
| LUT      | 9.3 k                   | 9.4 k                           |

Note: with the v7 `bundle=mem_q` / `bundle=mem_k` split on the top-level AXI,
HLS instantiates two conv implementations (`gdn_depthwise_conv_silu_1` for
the call producing `q`, `_2` for `k` / `v`). Their `conv_compute` schedules
differ slightly (different tap-DSP packing), which is why the DSP counts
diverge (36 vs 6).

## Why the original was so slow

The v1 inner loop was:

```c
conv_kern: for (k = 0; k < kernel_size; ++k) {
    if (source_row >= 0) {
        sum += in[...] * weights[...];
    }
}
```

with `sum` as a scalar accumulator. The carried fadd dep gave II=3, but more
importantly the *parent* `conv_col` was non-pipelined, so each col paid the
full 84-cycle conv_kern latency. Trip 4 M × 181 cyc = 759 M cycles.

The v2 attempt unrolled `conv_kern` to expose 4 parallel MACs but forgot that
those 4 MACs each issue an `m_axi` read, contending for the single AR channel.
HLS gave up on pipelining `conv_col`.

The v3 sliding-window + buffered-weights approach moves all heavy memory
traffic onto on-chip BRAM, leaving only one `m_axi` access per cycle in each
phase — exactly what the AXI master handles.

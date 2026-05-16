# Output Norm + Gate (`gdn_output_norm_and_gate`)

**Location:** `gdn_model.cpp:1349`

## Overview

Per-head RMSNorm on the recurrent-attention output, followed by a SiLU-gated
multiplication:

```
For each (token, head):
    sum  = Σ_i  attn_head[i] * attn_head[i]
    rms  = sqrt(sum / head_dim + eps)
    For each i:
        attn_head[i] = (attn_head[i] / rms * weight[i]) * gate[i] * sigmoid(gate[i])
```

Compile-time bounds: `num_heads = 8`, `head_dim = 256`, so each (token, head)
slice is 256 FP32 values.

## Optimisation 1: Pre-load shared norm weight once

The norm weight (`weight[head_dim]`, 256 floats) is the same for every
(token, head) pair. Pre-load it at the top of the function:

```c
float weight_loc[GDN_DV];
#pragma HLS array_partition variable=weight_loc cyclic factor=8

onorm_load_w: for (windex = 0; windex < head_dim; ++windex) {
#pragma HLS pipeline II=1
    weight_loc[windex] = weight[windex];
}
```

Cost: 256 floats = 1 KB on chip. Eliminates `head_dim × num_heads × num_tokens`
m_axi reads (e.g., 256 × 8 × 2048 = 4.2 M reads) across a single call.

The cyclic factor=8 partition matches the multi-bank read pattern in the
gate-write phase.

## Optimisation 2: Per-(token, head) on-chip buffers

The original implementation read `attn_head[i]` from m_axi twice — once in the
sum-of-squares pass, once in the gate pass — and wrote it back in the gate
pass. This made the gate pass an m_axi read followed by an m_axi write of the
same address every iteration, which HLS treats as a carried dep on the `gmem`
bus:

```
HLS 200-880  Unable to enforce a carried dependence constraint (II=160) between
             bus response on gmem (line 786) and bus request on gmem (line 784)
```

with II=160 — essentially serialised.

The current implementation pulls `attn_head[256]` and `gate_head[256]` into
local arrays once per (token, head):

```c
float attn_loc[GDN_DV];
float gate_loc[GDN_DV];
#pragma HLS array_partition variable=attn_loc cyclic factor=8
#pragma HLS array_partition variable=gate_loc cyclic factor=8
```

Cost: 2 KB on chip per call site. With `attn_loc` and `weight_loc` available
locally, the gate pass only writes to `attn_head[i]` (no read), so there is no
intra-iteration m_axi hazard.

Per (token, head) the work splits into three pipelined sub-phases:

| Phase            | I/O                                  | Trip | II |
|------------------|--------------------------------------|-----:|---:|
| `onorm_sq`       | m_axi read attn → local + square     | 256  | 1  |
| `onorm_load_g`   | m_axi read gate → local              | 256  | 1  |
| `onorm_gate`     | local read + m_axi write attn        | 256  | 1  |

## Optimisation 3: Tree reduce for sum-of-squares

The original used a `double sum` scalar accumulator. The carried fadd dep (FP64
fadd is 7-cycle pipelined) forced II=3 in `onorm_sq`. v1 switched to FP32 (the
final value gets cast to float anyway) and to 8-lane partial accumulators
keyed by `index & 7`, but HLS muxed the 8-element `onorm_partial` array into a
single mux register and tracked the dep on the mux — still II=2.

The current implementation stores per-element squared products into a
fully-partitioned scratch array and reduces them in a separate phase via the
helper `gdn_tree_reduce_256()` (an inline 8-level paired-sum tree, 256→128→
…→1). With no carried dep, `onorm_sq` pipelines at II=1.

```c
float sq_arr[GDN_DV];
#pragma HLS array_partition variable=sq_arr complete

onorm_sq: for (index = 0; index < head_dim; ++index) {
#pragma HLS pipeline II=1
    float v = attn_head[index];
    attn_loc[index] = v;
    sq_arr[index] = v * v;
}

float sum = gdn_tree_reduce_256(sq_arr);
float scale = 1.0f / sqrtf(sum / (float)head_dim + eps);
```

`gdn_tree_reduce_256` is shared with `load_qk` (q_sq, k_sq) and `dot_alpha`
(α). See [recurrent_attention.md §6](recurrent_attention.md) for the helper
definition.

A naive `for j unroll: sum += arr[j]` does not help here — HLS unrolls but
emits a 256-deep linear adder chain instead of a balanced tree. v5 made that
mistake; v6 introduced the explicit tree helper.

## Synthesis Results (U55C @ 100 MHz)

From `solution2/syn/report/csynth.rpt` for the v7 design.

Per (token, head) latency:

| Phase           | v0 (orig) | v4 (lane partial) | v7 (tree, U55C) |
|-----------------|----------:|------------------:|----------------:|
| onorm_sq        | 846       | 587 (II=2)        | **330** (II=1) |
| onorm_load_g    | n/a       | 327 (II=1)        | 329 |
| onorm_gate      | 40,962    | 344 (II=1)        | 342 |
| **total / call**| 41,808    | 1,258             | **1,052** (incl. tree reduce) |

For a full 2048-token sequence × 8 heads = 16,384 calls:

| Pass | Total `gdn_output_norm_and_gate` cycles | ns @ 100 MHz |
|------|---------------------------------------:|-------------:|
| v0   | 685.6 M | 6.86 s |
| v4   | 21.3 M  | 213 ms |
| v7   | **17.24 M** | **172 ms** |

That's **40× faster** than v0 and **19 % faster** than v4.

## Resource Cost (U55C)

| Resource | Value |
|----------|------:|
| BRAM_18K | 0     |
| DSP      | 13 (~0 %) |
| FF       | 26.5 k (~0 %) |
| LUT      | 17.6 k (1 %) |

No BRAM is needed: the per-(token, head) `attn_loc[256]` and `gate_loc[256]`
buffers fit in registers because of the cyclic factor=8 partitioning, and the
shared `weight_loc[256]` is similarly mapped. The DSP cost is small because
the helper tree reduce reuses fadd cores across the four call sites
(q_sq, k_sq, α, sum).

## Why the original was so slow

Two compounding problems:

1. **Read-then-write on the same m_axi address** in the gate loop produced an
   AXI carried dep that HLS could not break, forcing II=160 (effectively a
   serialised loop with full pipeline depth per iter).
2. **`double` accumulator** in the sum-of-squares loop produced an FP64 fadd
   carried dep that forced II=3.

Both are gone. The remaining cost is dominated by the three 256-iter
sub-passes, each at II=1, plus a one-shot tree reduce.

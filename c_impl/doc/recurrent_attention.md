# Optimised Recurrent Attention (`gdn_recurrent_attention`)

**Location:** `gdn_model.c:559`

## Overview

This module implements the gated delta rule recurrence, the core mechanism
that distinguishes GatedDeltaNet from standard linear attention. Each head
maintains a persistent state matrix S (256 x 256 FP32) that is updated at
every token step.

The implementation applies three optimisations from Gupta et al., "A
Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode
on FPGA":

1. **Persistent on-chip state** in BRAM
2. **Fused two-pass pipeline** (algebraic fusion)
3. **Column parallelism** P_K = 16

## Gated Delta Rule Algorithm

For each token t and head h:

```
Inputs: q, k, v (from projections), a (decay input), b (gate input)
State:  S[h] (d_k x d_v matrix, persistent across tokens)

1. Normalise: q_norm = q / ||q||,  k_norm = k / ||k||
2. Compute gates:
     beta = sigmoid(b)                    -- update gate
     g = exp(-exp(a_log) * softplus(a + dt_bias))  -- decay factor
3. Retrieval: r = S^T * k_norm
4. Delta:     dv = beta * (v - g * r)
5. Output:    o = (1/sqrt(d_k)) * S_new^T * q_norm
            = q_scale * (g * S_old^T * q_norm + alpha * dv)
   where     alpha = q_norm^T * k_norm
6. Update:    S = g * S + k_norm * dv^T
```

## Compile-Time Constants

```c
#define GDN_HEADS   8     // number of attention heads
#define GDN_DK    256     // query/key head dimension
#define GDN_DV    256     // value head dimension
#define GDN_PK     16     // column parallelism factor
```

These are used in C code. HLS pragmas use literal `16` because Vitis HLS
pragma processing does not expand C preprocessor macros.

## Optimisation 1: Persistent On-Chip State

### Problem
The naive implementation stores the 8 x 256 x 256 state matrix in external
DRAM (via `m_axi`). Each token step requires 4 full state matrix traversals
(decay, retrieval, update, output query), each reading/writing 256 x 256 =
65,536 FP32 values through the AXI bus. At ~100 cycles per AXI transaction,
this creates massive memory-bound latency.

### Solution
```c
static float state[GDN_HEADS][GDN_DK][GDN_DV];
#pragma HLS bind_storage variable=state type=RAM_2P impl=BRAM
```

The state is declared `static` so it persists across function calls (across
layers and tokens). It is bound to dual-port BRAM, providing one read port
and one write port per cycle. Total storage: 8 x 256 x 256 x 4 bytes = 2 MB,
consuming 938 BRAM_18K (69% of one SLR on VU11P).

The `recurrent_state` and `head_buffer` m_axi ports are kept in the interface
for API compatibility but are unused.

### State Clear
The state is cleared at the start of each forward pass (new sequence). The
clear loop is parallelised by P_K:

```c
state_clr_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
    #pragma HLS pipeline II=1
    for (pp = 0; pp < GDN_PK; ++pp) {
        #pragma HLS unroll
        state[h][j][i + pp] = 0.0f;
    }
}
```

This writes 16 elements per cycle, clearing all 524,288 elements in ~32,770
cycles (0.33 ms at 100 MHz).

## Optimisation 2: Fused Two-Pass Pipeline

### Problem
The naive algorithm requires 4 separate passes over the state matrix per token
per head:
1. **Decay pass:** `S[j][i] *= g` (read + write, 256x256)
2. **Retrieval pass:** `r[i] = sum_j S[j][i] * k[j]` (read, 256x256)
3. **Update pass:** `S[j][i] += k[j] * dv[i]` (read + write, 256x256)
4. **Output pass:** `o[i] = sum_j S[j][i] * q[j]` (read, 256x256)

### Solution: Algebraic Fusion
The key insight is that the output can be computed from the *old* state without
needing to read the *updated* state:

```
S_new = g * S_old + k_norm * dv^T
o = q_scale * S_new^T * q_norm
  = q_scale * (g * S_old^T * q_norm  +  (q_norm^T * k_norm) * dv)
  = q_scale * (g * o_buf           +  alpha          * dv)
```

This allows fusing passes 1+2+4 into a single read pass, and pass 3+1 into a
single read-modify-write pass:

**Fused Read Pass (Phase 2):**
```c
fused_rd_j: for (j = 0; j < GDN_DK; ++j) {
    float kj = k_loc[j], qj = q_loc[j];
    fused_rd_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
        #pragma HLS pipeline II=1
        for (pp = 0; pp < GDN_PK; ++pp) {
            #pragma HLS unroll
            float s = state[head_index][j][i + pp];
            r_buf[i + pp] += s * kj;    // retrieval
            o_buf[i + pp] += s * qj;    // partial output
        }
    }
}
```

**Delta Correction (Phase 3):**
```c
delta_out: for (i = 0; i < GDN_DV; ++i) {
    float d = beta * (v_loc[i] - g * r_buf[i]);
    dv[i] = d;
    out_head[i] = q_scale * (g * o_buf[i] + alpha * d);
}
```

Note: the decay factor `g` is applied to `r_buf` in the delta correction
(`v - g*r`), not during the read pass. This is algebraically equivalent to
decaying state before retrieval, but avoids modifying the state during the
read pass.

**Fused Write Pass (Phase 4):**
```c
fused_wr_j: for (j = 0; j < GDN_DK; ++j) {
    float kj = k_loc[j];
    fused_wr_i: for (i = 0; i < GDN_DV; i += GDN_PK) {
        #pragma HLS pipeline II=1
        for (pp = 0; pp < GDN_PK; ++pp) {
            #pragma HLS unroll
            state[head_index][j][i + pp] =
                g * state[head_index][j][i + pp] + kj * dv[i + pp];
        }
    }
}
```

This combines decay and rank-1 update into one read-modify-write pass.

### Result
Total state traversals reduced from 4 to 2 per token per head.

## Optimisation 3: Column Parallelism P_K = 16

### Mechanism
The innermost dimension of the state array (d_v = 256) is cyclically
partitioned by factor 16:

```c
#pragma HLS array_partition variable=state dim=3 cyclic factor=16
```

This creates 16 independent BRAM banks, each holding every 16th element. The
inner loops stride by `GDN_PK` with a fully unrolled inner loop of 16
iterations, allowing 16 parallel read or read-modify-write operations per
cycle.

### Local Buffer Partitioning
The per-head working buffers are also partitioned to match:

```c
#pragma HLS array_partition variable=r_buf cyclic factor=16
#pragma HLS array_partition variable=o_buf cyclic factor=16
#pragma HLS array_partition variable=dv    cyclic factor=16
#pragma HLS array_partition variable=v_loc cyclic factor=16
```

### Effective Throughput
- Fused read pass: 256 rows x 16 inner iterations = 4,096 + overhead ~ 4,103 cycles
- Fused write pass: same structure, ~ 4,103 cycles
- Per-head iteration total: ~10,712 cycles
- vs naive per-head: ~19,105,698 cycles

## Per-Head Pipeline Summary

```
Phase                    Cycles     Notes
-----------              ------     -----
load_qk                  849        Load Q, K from DRAM, compute L2 norms
load_v                   329        Load V from DRAM
norm_qk                  259        Normalise Q, K in local buffers
log (softplus)           11         Compute decay (pipelined log)
dot_alpha                517        alpha = q_norm^T * k_norm
init_ro                  18         Zero r_buf, o_buf (unrolled x16)
fused_rd_j/i             4,103      Fused read: retrieval + partial output
delta_out                371        Delta correction + output write
fused_wr_j/i             4,103      Fused write: decay + state update
-----------              ------
Total per head           ~10,712
x 8 heads x T tokens     85,696 .. 175,882,240
```

## Synthesis Results (Single-Layer, VU11P @ 100 MHz)

### Latency Comparison

| Metric | Naive | Optimised | Speedup |
|--------|-------|-----------|---------|
| recurrent_attention max cycles | 313.0 x 10^9 | 175.9 x 10^6 | **1,779x** |
| recurrent_attention max time | 3,130 sec | 1.759 sec | **1,779x** |
| Per-head iteration | 19.1 x 10^6 | 10,712 | **1,784x** |
| Top-level (gdn_attn_forward) | 469.6 x 10^9 | 191.0 x 10^9 | **2.46x** |

The 1,779x speedup on recurrent attention is due to eliminating external
memory traffic. The top-level speedup is 2.46x because `gdn_matmul` (26.8B
cycles) still dominates overall latency.

### Resource Comparison (recurrent_attention only)

| Resource | Naive | Optimised | Delta |
|----------|-------|-----------|-------|
| BRAM_18K | 0 | 938 | +938 (on-chip state) |
| DSP | 71 | 168 | +97 (16-wide parallel MACs) |
| FF | 21,047 | 43,426 | +22,379 |
| LUT | 59,181 | 84,242 | +25,061 |
| URAM | 0 | 0 | -- |

### Resource Comparison (top-level gdn_attn_forward)

| Resource | Naive | Optimised | Delta |
|----------|-------|-----------|-------|
| BRAM_18K | 0 | 938 | +938 |
| DSP | 228 | 317 | +89 |
| FF | 76,621 | 96,071 | +19,450 |
| LUT | 157,821 | 172,549 | +14,728 |

BRAM utilisation is 69% of one SLR (938/1344). The design fits within a
single SLR on VU11P and uses 23% of total device BRAM.

## Known Limitations

1. **II violation on `delta_out`:** The output write loop achieves II=4
   (not II=1) due to limited AXI memory ports for writing results back to
   DRAM. This is a minor overhead since the loop is only 256 iterations.

2. **Single-head sequential processing:** Heads are processed sequentially.
   Multi-head parallelism would require replicating the state BRAM (2 MB per
   head), which would exceed single-SLR BRAM capacity.

3. **HLS pragma macro limitation:** Vitis HLS pragmas do not expand C
   preprocessor macros, so `factor=16` is hardcoded as a literal in all
   `array_partition` and `unroll` pragmas. The `GDN_PK` define is still used
   in C code for loop bounds and stride.

## Parity Verification

- CSim parity: max_abs_diff = 2.62e-06, 0 failures (atol=1e-3, rtol=1e-3)
- Full model parity (test_parity.sh): PASS across all benchmarks (PiQA,
  HellaSwag, WinoGrande, ARC, BoolQ, WikiText, etc.), max diffs in 1e-4
  to 1e-6 range.

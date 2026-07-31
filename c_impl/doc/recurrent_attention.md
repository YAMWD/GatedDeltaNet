# Optimised Recurrent Attention (`gdn_recurrent_attention`)

**Status:** Active decode compute block. The prefill synthesis tables in this
document are historical.

**Location:** `gdn_model.cpp` (`gdn_recurrent_attention`, static helper)

## Overview

This module implements the gated delta rule recurrence, the core mechanism
that distinguishes GatedDeltaNet from standard linear attention. Each head
maintains a persistent state matrix S (256 × 256 FP32) that is updated at
every token step.

The implementation applies the following optimisations, in roughly the order
they were introduced:

1. **Partitioned BRAM working state** with one HBM restore/save per layer call
2. **Fused two-pass pipeline** (algebraic fusion → 4 state passes → 2)
3. **Column parallelism** P_K = 16 (16 MACs per cycle on state ops)
4. **`delta_out` + `delta_drain` split** (lifts II=16 → II=1)
5. **Tree-reduce reductions for `q_sq`, `k_sq`, `α`** (lifts II=2/3 → II=1)
6. **q/k AXI bundle split** (lifts `load_qk` II=2 → II=1)

(1)–(3) are inherited from Gupta et al., "A Persistent-State Dataflow
Accelerator for Memory-Bound Linear Attention Decode on FPGA". (4)–(6) were
added during the v1–v7 optimisation passes documented in
[optimization_log.md](optimization_log.md).

## Gated Delta Rule Algorithm

For each token *t* and head *h*:

```
Inputs: q, k, v (from projections), a (decay input), b (gate input)
State:  S[h] (d_k × d_v matrix, persistent across tokens)

1. Normalise: q_norm = q / ||q||,  k_norm = k / ||k||
2. Compute gates:
     beta = sigmoid(b)                                    (update gate)
     g = exp(-exp(a_log) * softplus(a + dt_bias))         (decay factor)
3. Retrieval / partial output (fused read pass):
     r[i]  = Σ_j S[j][i] * k_norm[j]
     o[i]  = Σ_j S[j][i] * q_norm[j]
4. α      = q_norm^T * k_norm                              (scalar)
5. Δv[i]  = β * (v[i] − g * r[i])
6. out[i] = (1/√d_k) * (g * o[i] + α * Δv[i])
7. State update (fused write pass):
     S[j][i] = g * S[j][i] + k_norm[j] * Δv[i]
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

## Optimisation 1: Partitioned BRAM Working State

### Problem
The naive implementation stores the 8 × 256 × 256 state matrix in external
DRAM (via `m_axi`). Each token step requires 4 full state matrix traversals
(decay, retrieval, update, output query), each reading/writing 256 × 256 =
65,536 FP32 values through the AXI bus. At ~100 cycles per AXI transaction,
this creates massive memory-bound latency.

### Solution
```c
static float state[GDN_HEADS][GDN_DK][GDN_DV];
#pragma HLS bind_storage variable=state type=RAM_2P impl=BRAM
#pragma HLS array_partition variable=state dim=3 cyclic factor=16
```

The state is declared `static`, bound to dual-port BRAM, and partitioned into
16 column banks. In the current decode-only kernel, each layer first restores
its 2 MiB slice from the 48 MiB HBM-resident `recurrent_state`, performs the
single-token update on BRAM, then saves the slice for the next kernel call.
The GPU-generated `.gdnstate` supplies the initial value after prompt prefill.

The BRAM copy removes external traffic from the inner gated-delta loops, but
state restore/save remains a material serial cost per token. `head_buffer` is
not used by recurrent attention; the top-level kernel uses it for persistent
Q/K/V convolution tails.

### State Restore and Save

Restore and save traverse `[head][k][v]` in contiguous 16-float groups. The
`v` dimension partition and Pack16 AXI access allow one 512-bit state transfer
per pipelined iteration. No state-clear path remains in the decode kernel.

## Optimisation 2: Fused Two-Pass Pipeline

### Problem
The naive algorithm requires 4 separate passes over the state matrix per
token per head:

1. **Decay pass:** `S[j][i] *= g`
2. **Retrieval pass:** `r[i] = Σ_j S[j][i] * k[j]`
3. **Update pass:** `S[j][i] += k[j] * dv[i]`
4. **Output pass:** `o[i] = Σ_j S[j][i] * q[j]`

### Solution: Algebraic Fusion
The output can be computed from the *old* state without needing the *updated*
state:

```
S_new = g * S_old + k_norm * dv^T
o = q_scale * S_new^T * q_norm
  = q_scale * (g * S_old^T * q_norm  +  (q_norm^T * k_norm) * dv)
  = q_scale * (g * o_buf            +  α                  * dv)
```

This fuses passes 1+2+4 into a single read pass and pass 3+1 into one
read-modify-write pass:

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

The innermost dimension of the state array (`d_v = 256`) is cyclically
partitioned by factor 16:

```c
#pragma HLS array_partition variable=state dim=3 cyclic factor=16
```

This creates 16 independent BRAM banks. The inner loop strides by `GDN_PK`
with a fully unrolled inner loop of 16 iterations, allowing 16 parallel read
or read-modify-write operations per cycle.

The per-head working buffers are partitioned to match:
```c
#pragma HLS array_partition variable=r_buf cyclic factor=16
#pragma HLS array_partition variable=o_buf cyclic factor=16
#pragma HLS array_partition variable=dv    cyclic factor=16
#pragma HLS array_partition variable=v_loc cyclic factor=16
```

## Optimisation 4: `delta_out` + `delta_drain` Split

The original `delta_out` loop wrote 16 P_K-parallel results directly to the
m_axi output port:

```c
delta_out: for (i = 0; i < GDN_DV; ++i) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=16
    out_head[i] = q_scale * (g * o_buf[i] + α * dv[i]);    // m_axi write!
}
```

HLS's "limited memory ports" check rejected 16 simultaneous AXI stores on
the shared `gmem` port and reduced the schedule to II=16.

Splitting into a compute-only phase (writes to on-chip `out_loc[256]`) plus
a drain phase (sequential m_axi store) keeps both phases at II=1:

```c
float out_loc[GDN_DV];
#pragma HLS array_partition variable=out_loc cyclic factor=16

delta_out: for (i = 0; i < GDN_DV; ++i) {
#pragma HLS pipeline II=1
#pragma HLS unroll factor=16
    float d = beta * (v_loc[i] - g * r_buf[i]);
    dv[i] = d;
    out_loc[i] = q_scale * (g * o_buf[i] + alpha * d);
}

delta_drain: for (i = 0; i < GDN_DV; ++i) {
#pragma HLS pipeline II=1
    out_head[i] = out_loc[i];
}
```

Per-call: 371 cyc (II=16) → 31 + 327 = 358 cyc (II=1 + II=1 sequential).

## Optimisation 5: Tree-Reduce for `q_sq`, `k_sq`, `α`

The reductions inside the per-token body had three forms in earlier versions:

| Quantity | v0 (orig) | v1 (lane partial) | v6/v7 (tree) |
|----------|-----------|-------------------|--------------|
| `q_sq`, `k_sq` | `double` scalar accumulator (II=3) | 8-lane FP32 partial (II=2, mux dep) | dedicated `qsq_arr[256]` + `gdn_tree_reduce_256` (II=1) |
| `α` (`q^T·k`)  | FP32 scalar (II=2) | 8-lane FP32 partial (II=2, mux dep) | `alpha_prod[256]` + `gdn_tree_reduce_256` (II=1) |

The store-products + tree-reduce pattern decouples the pipelined producer
loop (no carried dep) from the reduction (a one-shot tree). The reducing
helper:

```c
static float gdn_tree_reduce_256(const float arr[256]) {
#pragma HLS inline
    /* 8 explicit levels: 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1 */
    /* Each level fully unrolled with array_partition complete. */
    ...
    return l2[0] + l2[1];
}
```

Two reasons for the explicit tree (rather than `for j unroll: sum += arr[j]`):

1. **HLS does not auto-tree-balance** an unrolled `+=` accumulator chain —
   it emits a 256-deep linear adder, which made `onorm_sq_reduce` 256 × 4 =
   1024 cycles in v5 before this fix.
2. **Unroll requires constant trip count.** `head_dim` is a function
   parameter, so an unroll directive on a `for (j < head_dim)` loop is
   silently ignored.

The helper is `inline` so it lives in the caller's pipeline scope and is
shared across `q_sq`, `k_sq`, `α`, and `sum` (in onorm).

## Historical Optimisation 6: q/k AXI Bundle Split

This split applied to the retired `gdn_attn_forward` top. Current `gdn_forward`
places Q and K on the shared `gmem_qkv` bundle; the text below is retained to
explain the v7 synthesis result.

`load_qk` reads `q_head[j]` and `k_head[j]` from m_axi in the same iteration:

```c
load_qk: for (j = 0; j < GDN_DK; ++j) {
#pragma HLS pipeline II=1
    float qj = q_head[j];     // m_axi read
    float kj = k_head[j];     // m_axi read on same gmem port → contention
    q_loc[j] = qj;
    k_loc[j] = kj;
    qsq_arr[j] = qj * qj;
    ksq_arr[j] = kj * kj;
}
```

With both ports on the default `gmem` bundle, HLS emits "200-885: limited
memory ports" and forces II=2.

In v7 we placed `q` and `k` on separate AXI bundles in `gdn_attn_forward`:

```c
#pragma HLS interface m_axi port=q depth=129024 offset=slave bundle=mem_q
#pragma HLS interface m_axi port=k depth=129024 offset=slave bundle=mem_k
```

so the two reads are issued on independent AXI master ports and `load_qk`
pipelines at II=1. Side effect: HLS instantiates separate matmul/conv
instances for callers using `mem_q` vs `mem_k`, doubling some resource
counts. Total resource utilisation stays under 25 %.

## Historical Per-Head Pipeline Summary (v7)

```
Phase                    Cycles    Notes
-----------              ------    -----
load_qk                  258       Pipelined II=1, FP32 squared scratch
load_v                   329       Load V from DRAM, II=1
norm_qk                  259       Apply 1/||q||, 1/||k||, II=1
log/softplus             ~10       Pipelined log(...) for decay
dot_alpha                259       Pipelined product scratch + tree, II=1
init_ro                  18        Zero r_buf, o_buf (unrolled ×16)
fused_rd_j/i             4 103     Fused read: retrieval + partial output
delta_out                31        Compute Δv + out_loc (II=1, ×16)
delta_drain              327       Drain out_loc → m_axi (II=1)
fused_wr_j/i             4 103     Fused write: decay + state update
-----------              ------
Total per (token,head)   ~9 598 (was 19,105,698 in naive baseline)
```

For 2048 tokens × 8 heads = **157.3 M cycles total** vs the naive baseline of
**313 G cycles** — a **~2,000× speedup** on the recurrent module alone.

## Historical Synthesis Results (single-layer, U55C @ 100 MHz)

From `GDN_single_attn/solution2/syn/report/csynth.rpt`, target
`xcu55c-fsvh2892-2L-e`:

| Module                        | Naive   | v7      | Speedup |
|-------------------------------|--------:|--------:|--------:|
| `gdn_recurrent_attention` cycles | 313.0 G | 157.29 M | **~2,000×** |
| `gdn_recurrent_attention` time   | 3,130 s | 1.573 s  | **~2,000×** |
| Per-(token, head) iteration   | 19.1 M  | 9 598    | **~2,000×** |
| Top-level `gdn_attn_forward`  | 469.6 G | 141.03 G | **3.33×** |

## Historical Resource Cost (`gdn_recurrent_attention` v7)

| Resource | Naive | v7    | Δ       |
|----------|------:|------:|--------:|
| BRAM_18K | 0     | 258   | +258 (on-chip state) |
| DSP      | 71    | 494   | +423 |
| FF       | 21 k  | 92.7 k | +72 k |
| LUT      | 59 k  | 79.3 k | +20 k |
| URAM     | 0     | 0     | — |

On the U55C target, HLS maps each of the 16 cyclic-partitioned state banks
(`gdn_recurrent_attention_state_X_U`) into 16 BRAM_18K blocks (16 × 16 = 256,
plus a few control-path BRAMs ⇒ 258 total). The state matrix (8 × 256 × 256 ×
4 bytes = 2 MB) is the single-largest contributor to the layer's BRAM budget.

## Known Limitations

1. **Single-head sequential processing** — heads are processed sequentially.
   Multi-head parallelism would require replicating the state BRAM (2 MB per
   head), exceeding single-SLR BRAM capacity for 8 heads.
2. **HLS pragma macro limitation** — Vitis HLS pragmas do not expand C
   preprocessor macros, so `factor=16` is hardcoded as a literal in all
   `array_partition` and `unroll` pragmas. The `GDN_PK` define is still used
   in C code for loop bounds and stride.
3. **State traffic is serialized with the rest of the layer.** Every token
   restores and saves 2 MiB per layer through the state master. Spreading or
   overlapping this traffic is the main non-GEMV architectural opportunity.

## Parity Verification

- Single-layer `gdn_attn_test`: max_abs_diff = 1.19 × 10⁻⁶, 0 failures
  (atol = 1 × 10⁻³, rtol = 1 × 10⁻³).
- Full-model `test_parity.sh` (v0 / v1 / v4 baselines): PASS across all
  benchmarks (PiQA, HellaSwag, WinoGrande, ARC, BoolQ, WikiText, ...) with
  max diffs in 1 × 10⁻⁴ to 1 × 10⁻⁶ range. v7 result pending on the same
  fixtures.

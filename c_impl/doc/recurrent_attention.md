# Optimised Recurrent Attention (`gdn_recurrent_attention`)

**Status:** Active Iter37 decode compute block. The prefill and v1--v7
synthesis discussions later in this document are historical.

**Location:** `gdn_model.cpp` (`gdn_recurrent_attention`, static helper)

## Overview

This module implements the gated delta rule recurrence. Each of the eight
heads maintains a persistent 256 x 256 FP32 state matrix. Iter37 combines four
features:

1. one head-local 256 x 256 URAM buffer rather than an all-head layer buffer;
2. algebraic fusion that reduces four state traversals to a retrieval pass and
   an update pass;
3. four external state stripes appended to weight ports 28--31; and
4. 32-column recurrent arithmetic supplied by two state ports at a time.

The external `.gdnstate` format remains contiguous
`[layer][head][K][V]`. The host scatters Pack16 words into four 12 MiB tails
when it uploads the state; the kernel updates that striped representation in
place across decode calls.

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

## Iter37 Compile-Time Constants

```c
#define GDN_HEADS   8     // number of attention heads
#define GDN_DK    256     // query/key head dimension
#define GDN_DV    256     // value head dimension
#define GDN_RECURRENT_LANES 32
#define GDN_RECURRENT_STATE_PORTS 4
#define GDN_RECURRENT_STATE_FIRST_PORT 28
```

`Pack16` remains the external 512-bit transfer unit. `GDN_RECURRENT_LANES=32`
means each fused loop consumes two Pack16 words from two independent ports and
performs 32 state-column updates per cycle.

## Head-Local State and Four-Port Layout

All 24 layers require 48 MiB, so the recurrent state cannot remain wholly on
chip. The kernel uses one head-local buffer:

```c
float state[256][256];
#pragma HLS bind_storage variable=state type=RAM_2P impl=URAM
#pragma HLS array_partition variable=state dim=2 cyclic factor=32
```

Every row contains 16 Pack16 words. Words 0--7 cover columns 0--127 and
alternate between ports 28 and 29; words 8--15 cover columns 128--255 and
alternate between ports 30 and 31. Consequently each port owns four contiguous
logical words per row and 1,024 words per head. The host appends each compact
stripe after the fixed 43,728,896-float GEMV shard boundary.

`fused_rd01` reads ports 28/29 together for the low half, then `fused_rd23`
reads ports 30/31 together for the high half. `fused_wr01` and `fused_wr23`
write the same layout. Each loop has a 1,024-iteration II=1 schedule and each
individual state port sees a monotonic burst. GEMV and recurrence are serial,
so reusing the four weight masters adds no competing memory operation.

## Fused Two-Pass Pipeline

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

The read loops capture every old-state word in the head-local URAM while also
accumulating `r` and partial `o`. The write loops consume that captured word,
apply decay plus the rank-one update, and immediately persist the updated
Pack16 pair. This removes the standalone full-layer restore/save passes that
preceded Iter36 without changing FP32 expression order.

Integrated Vitis HLS 2022.2 estimates 43,873--44,081 cycles per layer call,
down from Iter36's 76,713. All four state loops achieve II=1. The complete
kernel uses 3,627 DSPs and 48 URAMs, and the routed 100 MHz image measures
51.451 ms/token with exact 64-token parity.

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

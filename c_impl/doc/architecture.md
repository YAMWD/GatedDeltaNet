# GatedDeltaNet HLS Accelerator -- Architecture Reference

This document describes the Vitis HLS implementation of GatedDeltaNet-1.3B
inference in `c_impl/`. It is intended as a reference for future development
sessions.

## 1. Model Configuration (GatedDeltaNet-1.3B)

| Parameter        | Value |
|------------------|-------|
| Hidden dim       | 2048  |
| Num heads        | 8     |
| Head dim (Q/K)   | 256   |
| Value dim (V)    | 256   |
| Intermediate     | 5632  |
| Num layers       | 24    |
| Conv kernel size | 4     |
| Max seq len      | 2048  |
| Vocab size       | 32000 |

State matrix per layer: 8 heads x 256 x 256 FP32 = 2 MB.

## 2. File Layout

| File                    | Role |
|-------------------------|------|
| `gdn_model.h`          | Public API: structs (`GDNWeightHeader`, `GDNLayerWeights`, `GDNModel`, `GDNRunState`), function prototypes for full-model, single-layer, and matmul-only tops |
| `gdn_model.cpp`        | Main synthesizable implementation + HLS top functions (`gdn_forward`, `gdn_attn_forward`, `gdn_matmul_top`). Contains the systolic matmul kernel as in-file static helpers reused by all three tops. |
| `gdn_eval.cpp`        | Host testbench: loads `.gdnw` weights, reads `.gdnreq` fixtures, runs `gdn_forward`, writes JSON output |
| `gdn_attn_test.cpp`   | Host testbench for single-layer attention: loads `.gdnw` + `.gdnblk`, runs `gdn_attn_forward`, checks parity |
| `gdn_matmul_test.cpp` | Host testbench for the systolic matmul (`gdn_matmul_top`) against a native-C golden matmul |
| `host.cpp`            | XRT host program for on-card execution against `gdn_forward.xclbin` |
| `test.tcl`             | Vitis HLS TCL script for full-model (csim/csynth/cosim), targets Alveo U55C |
| `test_single_GDN_attn_synth.tcl` | Current TCL script for single-layer attention systolic synthesis, targets Alveo U55C |
| `test_single_GDN_attn.tcl` | Older v7 single-layer attention script, retained for the pre-systolic tiled-matmul baseline |
| `test_matmul.tcl`      | Matmul-only csim/csynth script (top = `gdn_matmul_top`); default test is 2048 x 2048 x 2048 |
| `test_parity.sh`       | Automated end-to-end parity test against Python golden results |
| `hw.cfg`, `pblock_pe_split.tcl` | v++ link configuration and pre-place floorplan TCL (SLR split for the U55C bitstream) |
| `Makefile`             | Builds host testbenches, v++ kernel (`xo`/`xclbin`), and the XRT host; `make run_hw` is the end-to-end on-card path |

### Weight and Fixture Formats

- **`.gdnw`** -- Flat binary: 60-byte `GDNWeightHeader` followed by all FP32 weights
  in layer order. Total ~5.6 GB for GDN-1.3B.
- **`.gdnreq`** -- Pretokenized evaluation fixture: header + int32 token IDs +
  golden log-probabilities/scores for parity checking.
- **`.gdnblk`** -- Single-layer attention fixture: header + input tensor + golden
  output tensor for one transformer block.

## 3. HLS Top Functions

### 3.1 `gdn_forward` (full model)

**Location:** `gdn_model.cpp:1432`

Full 24-layer GatedDeltaNet forward pass. All arguments are flat pointers with
`m_axi` interfaces. The function:

1. Embeds tokens (`gdn_embed_tokens`)
2. Loops over 24 layers, each performing:
   - RMSNorm on input
   - 6x matmul projections (Q, K, V, A, B, Gate)
   - Depthwise conv1d + SiLU on Q, K, V
   - Recurrent attention (gated delta rule)
   - Output norm + gate
   - Output projection matmul
   - Residual add
   - RMSNorm on residual
   - MLP: gate projection, up projection, SwiGLU, down projection
   - Residual add
3. Final RMSNorm

### 3.2 `gdn_attn_forward` (single-layer attention)

**Location:** `gdn_model.cpp:1665`

Single-layer attention forward for isolated synthesis/co-simulation. Uses the
same internal functions as `gdn_forward` but only runs one layer. This is the
primary synthesis target for optimisation work.

The struct-based wrapper `gdn_attn_forward_layer` is provided for testbench
convenience but cannot be the HLS top function (HLS does not support struct
pointers in top-function arguments).

## 4. Compute Submodules

### 4.1 `gdn_embed_tokens`

**Location:** `gdn_model.cpp:381`

Copies embedding rows from the weight table into the hidden state buffer.
Simple lookup, no compute.

### 4.2 `gdn_rmsnorm_rows`

**Location:** `gdn_model.cpp:408`

RMSNorm over each row (token) of the input. Uses `double` accumulation for
numerical stability. Called before every projection block and at the final
output.

### 4.3 `gdn_matmul_systolic` (Current Large-GEMM Engine)

**Location:** `gdn_model.cpp` (in-file static helper)

Current GEMM engine for large projections. One-chain 1-D systolic array:

```
16 PEs x 16 output columns/PE = 256 FP32 MAC/cycle peak
```

The kernel uses `hls::stream` and a `#pragma HLS dataflow` region around
`ReadA`, `ReadB`, a 16-PE chain, `SinkAB`, and `WriteC_chain`. It supports
runtime `num_rows`, pads rows on chip to a multiple of 16, and supports
`in_dim <= 5632` so the MLP down projection fits without host-side K tiling.
Called directly from `gdn_forward` / `gdn_attn_forward` and also exposed as
the standalone HLS top `gdn_matmul_top` (for `test_matmul.tcl` and
`gdn_matmul_test`). See [systolic_matmul.md](systolic_matmul.md).

### 4.4 `gdn_matmul_tiled` (Fallback)

**Location:** `gdn_model.cpp:866`

Legacy 16 x 16 x 16 tiled matrix multiplication. It remains in the current
design as the fallback for A/B projections where `out_dim=8`, which is too
small for the 16-column systolic output tile. Manual-flattened R x C compute
with an explicit balanced fadd tree gives II=1 in the inner pipeline. See
[tiled_matmul.md](tiled_matmul.md).

### 4.5 `gdn_depthwise_conv_silu`

**Location:** `gdn_model.cpp:1037`

Depthwise 1D convolution with SiLU activation. Kernel size is always 4
(causal, looking back 3 positions). Applied independently to Q, K, V after
their linear projections. Uses pre-buffered weights and a 4-row sliding window;
two-phase per-row execution (load + shift, then compute + write) keeps both
phases at II=1. See [depthwise_conv.md](depthwise_conv.md).

### 4.6 `gdn_recurrent_attention` (Optimised)

**Location:** `gdn_model.cpp:1129`

Core gated delta rule recurrence. Persistent BRAM state, fused two-pass
read/write, P_K=16 column parallelism, on-chip out_loc + drain split for
the AXI write phase, and tree-reduced L2 norm / α reductions. See
[recurrent_attention.md](recurrent_attention.md).

### 4.7 `gdn_output_norm_and_gate`

**Location:** `gdn_model.cpp:1349`

Per-head RMSNorm on attention output, followed by gated SiLU:
```
out[i] = RMSNorm(attn[i]) * gate[i] * sigmoid(gate[i])
```
Pre-loaded shared weight, on-chip per-(token, head) attn/gate buffers, and a
tree-reduced sum-of-squares give II=1 across all sub-passes. See
[output_norm.md](output_norm.md).

### 4.8 `gdn_swiglu_inplace`

**Location:** `gdn_model.cpp:1424`

Element-wise SwiGLU activation for MLP: `gate[i] = SiLU(gate[i]) * up[i]`.

### 4.9 `gdn_tree_reduce_256` (helper)

**Location:** `gdn_model.cpp:343`

Inline 8-level paired-sum FP32 fadd tree (256 → 128 → … → 1). Used by
`gdn_recurrent_attention` (`q_sq`, `k_sq`, `α`) and `gdn_output_norm_and_gate`
(`sum`) to reduce per-element scratch arrays. Replaces `for j unroll: sum +=
arr[j]` patterns, which HLS emits as a 256-deep linear adder rather than a
balanced tree.

## 5. Data Flow (Single Layer)

```
input (num_tokens x 2048)
  |
  +-- RMSNorm --> x_norm
  |
  +-- Systolic MatMul x4 --> Q, K, V, Gate
  +-- Tiled MatMul x2 ----> A, B             (small out_dim=8 fallback)
  |
  +-- DepthwiseConv1D+SiLU --> Q', K', V'   (causal conv, kernel=4)
  |
  +-- RecurrentAttention(Q', K', V', A, B)   (gated delta rule)
  |       |
  |       +-- L2 normalise Q, K per head
  |       +-- Compute decay g, gate beta per head
  |       +-- For each token x head:
  |       |     Phase 1: alpha = q_norm^T * k_norm
  |       |     Phase 2: fused read (retrieval r + partial output o)
  |       |     Phase 3: delta correction + output
  |       |     Phase 4: fused write (decay + state update)
  |       |
  |       +-- Output: attn (num_tokens x 2048)
  |
  +-- OutputNormGate(attn, Gate)
  |
  +-- Systolic MatMul(o_proj) --> tmp_hidden
  |
  +-- Residual: x += tmp_hidden
  |
  +-- RMSNorm --> x_norm
  |
  +-- MLP: Systolic MatMul(gate_proj), Systolic MatMul(up_proj),
  |        SwiGLU, Systolic MatMul(down_proj)
  |
  +-- Residual: x += tmp_hidden
  |
  v
output (num_tokens x 2048)
```

## 6. HLS Interface Strategy

All external data is accessed through AXI4 master (`m_axi`) ports with
`offset=slave` (base address set via AXI-Lite control registers). Scalar
arguments use `s_axilite`.

The `GDNWeightHeader` struct is read from DRAM via `m_axi` to extract config
parameters at runtime (hidden size, num heads, etc.). Weight pointers for
each layer are computed as offsets into the flat `weight_data` array.

### AXI bundle topology (`gdn_attn_forward`, current systolic path)

| Bundle    | Ports                                                                 |
|-----------|------------------------------------------------------------------------|
| `gmem`    | config, input, output, v, a, b, gate, attn, tmp_hidden, recurrent_state, head_buffer |
| `mem_weights` | weight_data |
| `mem_q`   | q                                                                      |
| `mem_k`   | k                                                                      |

Splitting `q` and `k` onto dedicated bundles lets `gdn_recurrent_attention`'s
`load_qk` issue both reads in the same cycle (II=1). The AXI port-contention
warning ("HLS 200-885: limited memory ports") was the only structural
violation that could not be resolved without this bundle split.

`weight_data` is also placed on `mem_weights` so the systolic `ReadB` task
does not contend with `ReadA`, which reads activations from the default bundle.
The full-model `gdn_forward` top uses `mem_weights` for weights and the default
bundle for all activation/state buffers; only `gdn_attn_forward` splits q/k
because those are explicit top-level buffers in the single-layer top.

## 7. Numerical Precision

- All compute is FP32.
- L2 norm and RMSNorm accumulators use `double` (FP64) for numerical stability.
- Parity target: < 1e-3 absolute tolerance vs Python golden reference.
- Observed parity: ~1e-5 to 1e-6 max absolute difference.

## 8. Target Device

The canonical target for both `gdn_forward` and `gdn_attn_forward` is the
Xilinx Alveo U55C card:

| Device | FPGA | TCL |
|--------|------|-----|
| `xcu55c-fsvh2892-2L-e` | Virtex UltraScale+ VU13P (U55C card) | `test.tcl`, `test_single_GDN_attn.tcl` |

Clock period: 10 ns (100 MHz).

## 9. Current Synthesis Snapshot

### Single-layer attention

| Metric | v7 tiled matmul | Current systolic matmul |
|--------|----------------:|------------------------:|
| Top-level latency | 141.03 G cycles | 3.976 G cycles |
| Timing slack @ 100 MHz target | 0.00 ns | -0.04 ns |
| BRAM_18K | 322 (7 %) | 1602 (39 %) |
| DSP | 1042 (11 %) | 4690 (51 %) |
| FF | 209.8 k (8 %) | 848.9 k (32 %) |
| LUT | 237.0 k (18 %) | 932.0 k (71 %) |

The latency reduction comes from replacing the dominant large projection
matmuls with the systolic dataflow kernel. The resource increase is expected:
the current single-attention top contains multiple systolic dataflow instances
plus the persistent recurrent state. The report has a small timing miss
(-0.04 ns).

### Full model

| Metric | Current `gdn_forward` |
|--------|----------------------:|
| Top-level latency | 129.686 G cycles |
| Timing slack @ 100 MHz target | -0.04 ns |
| BRAM_18K | 1058 (26 %) |
| DSP | 2847 (31 %) |
| FF | 508.4 k (19 %) |
| LUT | 580.3 k (44 %) |

The full model does not instantiate 24 physical layers. The layer loop is a
time loop, so resources are for one reused datapath and the 24 layers increase
latency rather than multiplying resource use.

## 10. Optimisation History

The v1-v7 optimisation passes applied to `gdn_attn_forward` before the
systolic matmul rewrite are documented in [optimization_log.md](optimization_log.md).
The v7 tiled-matmul baseline for single-layer attention was:

| Metric                         | Baseline (v0) | Final (v7) | Δ |
|--------------------------------|---------------|-----------:|---|
| Top-level latency              | 190.96 G cyc  | **141.03 G** | −26 % |
| Outstanding II violations      | 7             | **0**        | −7 |
| Timing slack @ 100 MHz target  | −0.46 ns      | **0.00 ns**  | +0.46 ns (closes timing) |
| BRAM_18K                       | 938 (23 %)    | 322 (7 %)    | −616 |
| DSP                            | 317 (3 %)     | 1042 (11 %)  | +725 |
| LUT                            | 172 k (13 %)  | 237 k (18 %) | +65 k |
| FF                             | 96 k (3 %)    | 210 k (8 %)  | +114 k |

(Numbers from `GDN_single_attn/solution2/syn/report/csynth.rpt`, U55C target.
The BRAM count drops on U55C because HLS maps the persistent state with a
denser per-partition allocation than it did on the prior VU11P run.)

The current architecture implements that structural follow-up through the
systolic matmul described in [systolic_matmul.md](systolic_matmul.md). The
older tiled matmul documentation remains relevant for the A/B projection
fallback path and for explaining the pre-systolic baseline.

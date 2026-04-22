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
| `gdn_model.h`          | Public API: structs (`GDNWeightHeader`, `GDNLayerWeights`, `GDNModel`, `GDNRunState`), function prototypes for full-model and single-layer forward |
| `gdn_model.c`          | All synthesizable compute functions + HLS top functions (`gdn_forward`, `gdn_attn_forward`) |
| `gdn_eval.c`           | Host testbench: loads `.gdnw` weights, reads `.gdnreq` fixtures, runs `gdn_forward`, writes JSON output |
| `gdn_attn_test.c`      | Host testbench for single-layer attention: loads `.gdnw` + `.gdnblk`, runs `gdn_attn_forward`, checks parity |
| `test.tcl`             | Vitis HLS TCL script for full-model (csim/csynth/cosim) |
| `test_single_GDN_attn.tcl` | TCL script for single-layer attention (csim/csynth/cosim), targets Alveo U55C |
| `test_opt_attn.tcl`    | TCL script for optimized single-layer attention (csim/csynth only), targets VU11P |
| `test_parity.sh`       | Automated end-to-end parity test against Python golden results |
| `Makefile`             | Native C build (GCC, no BLAS) |

### Weight and Fixture Formats

- **`.gdnw`** -- Flat binary: 60-byte `GDNWeightHeader` followed by all FP32 weights
  in layer order. Total ~5.6 GB for GDN-1.3B.
- **`.gdnreq`** -- Pretokenized evaluation fixture: header + int32 token IDs +
  golden log-probabilities/scores for parity checking.
- **`.gdnblk`** -- Single-layer attention fixture: header + input tensor + golden
  output tensor for one transformer block.

## 3. HLS Top Functions

### 3.1 `gdn_forward` (full model)

**Location:** `gdn_model.c:800`

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

**Location:** `gdn_model.c:1030`

Single-layer attention forward for isolated synthesis/co-simulation. Uses the
same internal functions as `gdn_forward` but only runs one layer. This is the
primary synthesis target for optimisation work.

The struct-based wrapper `gdn_attn_forward_layer` is provided for testbench
convenience but cannot be the HLS top function (HLS does not support struct
pointers in top-function arguments).

## 4. Compute Submodules

### 4.1 `gdn_embed_tokens`

**Location:** `gdn_model.c:345`

Copies embedding rows from the weight table into the hidden state buffer.
Simple lookup, no compute.

### 4.2 `gdn_rmsnorm_rows`

**Location:** `gdn_model.c:372`

RMSNorm over each row (token) of the input. Uses `double` accumulation for
numerical stability. Called before every projection block and at the final
output.

### 4.3 `gdn_matmul` (Tiled)

**Location:** `gdn_model.c:407`

General-purpose tiled matrix multiplication. See [tiled_matmul.md](tiled_matmul.md)
for detailed documentation.

### 4.4 `gdn_depthwise_conv_silu`

**Location:** `gdn_model.c:529`

Depthwise 1D convolution with SiLU activation. Kernel size is always 4
(causal, looking back 3 positions). Applied independently to Q, K, V after
their linear projections. Each channel is convolved independently (depthwise).

### 4.5 `gdn_recurrent_attention` (Optimised)

**Location:** `gdn_model.c:559`

Core gated delta rule recurrence. See [recurrent_attention.md](recurrent_attention.md)
for detailed documentation.

### 4.6 `gdn_output_norm_and_gate`

**Location:** `gdn_model.c:755`

Per-head RMSNorm on attention output, followed by gated SiLU:
```
out[i] = RMSNorm(attn[i]) * gate[i] * sigmoid(gate[i])
```

### 4.7 `gdn_swiglu_inplace`

**Location:** `gdn_model.c:792`

Element-wise SwiGLU activation for MLP: `gate[i] = SiLU(gate[i]) * up[i]`.

## 5. Data Flow (Single Layer)

```
input (num_tokens x 2048)
  |
  +-- RMSNorm --> x_norm
  |
  +-- MatMul x6 --> Q, K, V, A, B, Gate     (projections)
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
  +-- MatMul(o_proj) --> tmp_hidden
  |
  +-- Residual: x += tmp_hidden
  |
  +-- RMSNorm --> x_norm
  |
  +-- MLP: MatMul(gate_proj), MatMul(up_proj), SwiGLU, MatMul(down_proj)
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

## 7. Numerical Precision

- All compute is FP32.
- L2 norm and RMSNorm accumulators use `double` (FP64) for numerical stability.
- Parity target: < 1e-3 absolute tolerance vs Python golden reference.
- Observed parity: ~1e-5 to 1e-6 max absolute difference.

## 8. Target Devices

| Device | FPGA | Use |
|--------|------|-----|
| `xcvu11p-flga2577-1-e` | Virtex UltraScale+ VU11P | Full model synthesis, optimised attention |
| `xcu55c-fsvh2892-2L-e` | Alveo U55C | Single-layer attention (naive baseline) |

Both targets use a 10 ns clock period (100 MHz).

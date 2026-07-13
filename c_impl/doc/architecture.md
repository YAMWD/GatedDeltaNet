# GatedDeltaNet Decode Accelerator Architecture

**Status:** Current production architecture as of 2026-07-13.

The accelerator in `c_impl/` is decode-only. Prompt prefill runs on the GPU,
which exports a fixed-size recurrent and convolution state. The U55C then
forwards one token per kernel invocation. Large projections use an eight-port
HBM GEMV engine; the retired prefill GEMM and systolic matmul are not part of
`gdn_forward`.

## Model Configuration

| Parameter | Value |
|---|---:|
| Hidden dimension | 2,048 |
| Attention heads | 8 |
| Q/K/V head dimension | 256 |
| MLP intermediate dimension | 5,632 |
| Layers | 24 |
| Convolution kernel | 4 |
| Vocabulary | 32,000 |

The recurrent state is `24 x 8 x 256 x 256` FP32 values, or 48 MiB. The
convolution tail stores three prior rows for Q, K, and V in every layer, about
1.7 MiB total. Both are persistent HBM buffers across kernel calls.

## Source Layout

| Path | Role |
|---|---|
| `gdn_model.cpp`, `gdn_model.h` | Synthesizable decode kernel, host-side model loader, weight sharder, and native wrapper |
| `gdn_eval.cpp` | Native decode parity harness driven by `.gdnstate` and `.gdnreq` files |
| `host.cpp` | XRT host that allocates buffers, uploads the eight weight shards and state, and invokes `gdn_forward` |
| `hw.cfg` | U55C HBM bank bindings and implementation directives |
| `Makefile` | Native, `v++`, XRT host, and on-card targets |
| `microbench/gemv_tile/` | Standalone routed 32-port GEMV experiment; not yet integrated into `gdn_forward` |

The `.gdnw` file contains the original FP32 model weights. The host derives
eight compact large-projection shards from it. A `.gdnstate` file contains the
GPU-prefilled recurrent state, convolution tails, and initial decode token.

## Kernel Entry Point

`gdn_forward` is the only current HLS top. It requires `num_tokens == 1` and
performs a complete greedy decode step:

1. Embed the input token.
2. Run all 24 GatedDeltaNet layers against persistent state.
3. Apply final RMSNorm.
4. Run the sharded `lm_head` GEMV.
5. Compute argmax on chip and return the next token through `x_norm[0]`.

`gdn_decode_step_host` calls the same function for native C++ testing. There is
no prefill mode, `decode_flags`, `gdn_attn_forward`, or matmul HLS top in the
current API.

## Per-Layer Data Flow

```text
token/residual x
  -> RMSNorm
  -> GEMV: q, k, v, gate
  -> tiny GEMV: a, b (8 outputs each)
  -> persistent-tail depthwise conv + SiLU on q, k, v
  -> recurrent gated-delta attention (restore/update/save state)
  -> output norm + gate
  -> GEMV: output projection
  -> residual add
  -> RMSNorm
  -> GEMV: MLP gate and up projections
  -> SwiGLU
  -> GEMV: MLP down projection
  -> residual add
```

Each layer therefore performs eight large `gdn_gemv` calls. The A/B
projections use `gdn_gemv_tiny` because their output dimension is only eight;
they do not use a retired tiled or systolic matmul.

## Integrated GEMV Engine

The production `gdn_gemv` splits output rows across `GEMV_CHANNELS=8` compact
weight shards. Each shard is attached to an independent 512-bit AXI master and
holds the same projection sequence:

```text
q, k, v, gate, o, mlp_gate, mlp_up, mlp_down, then lm_head
```

For a projection with shape `[out_dim, in_dim]`, channel `c` stores rows
`[c * out_dim/8, (c + 1) * out_dim/8)`. The projection has the same Pack16
offset in every shard, so all readers can start together.

The HLS dataflow region contains:

1. `gemv_pe_bcast`: reads the activation vector once and broadcasts each
   Pack16 beat to eight activation streams.
2. Eight `gemv_read_ch` processes: stream contiguous weight stripes from the
   independent HBM masters at a target II of one 512-bit beat per cycle.
3. Eight `gemv_pe_mac` processes: load private activation copies, perform 16
   FP32 multiplies per weight beat, and reduce them with an adder tree.
4. `gemv_collect`: packs scalar dot products into 512-bit output writes while
   preserving `out[channel * stripe + row]` ordering.

`gemv_pe_mac` processes eight packs in an unrolled group every eight cycles.
Eight independent partial sums hide FP32 adder latency, while ping-pong row
buffers allow the previous row to drain without stopping the next row. The
flattened row/reduction loop pays pipeline startup once per channel instead of
once per output row.

The activation is intentionally copied once per PE. A single shared array
would require enough read ports and global fanout to feed all eight engines,
which would move the routing problem into the activation network.

## Persistent State

The recurrent state and convolution tails remain in HBM between token calls.
Within each layer:

- Q/K/V convolution restores three prior rows, computes the current row, and
  saves the updated tail.
- `gdn_recurrent_attention` restores that layer's 2 MiB state into partitioned
  BRAM, updates it for one token, and writes it back.

This makes decode work and memory independent of context length. The state is
not permanently resident in BRAM across kernel invocations.

## HBM Mapping

`hw.cfg` assigns non-overlapping U55C bank groups:

| Banks | Data |
|---|---|
| HBM[0] | Configuration, tokens, A/B activations, recurrent state, convolution tails |
| HBM[1] | Residual, normalized activation, temporary hidden |
| HBM[2] | Q/K/V, gate, attention |
| HBM[3] | MLP intermediates and logits |
| HBM[4:19] | Eight GEMV shards, two pseudo-channels each |
| HBM[20:30] | Original weight blob for embeddings, norms, conv, and tiny projections |
| HBM[31] | Free |

This is 13 AXI masters: eight GEMV shard masters, one full-weight master, three
activation masters, and one scalar/control master. The original weight blob is
still required because only the large projection matrices are copied into the
compact shards.

## Correctness and Measured Status

All arithmetic is FP32. Reduction order is kept stable where bit-exact decode
depends on it. The current integrated U55C design is bit-exact to the GPU
golden over 64 generated tokens and completes the full forward, `lm_head`, and
argmax at 121.4 ms/token. GEMV accounts for about 68 ms of that result.

The routed 32-port mono-kernel in `microbench/gemv_tile/` is a separate scaling
milestone. At an achieved 130.6 MHz it sustains 263.063 GB/s and 131.531
GFLOP/s with exact parity and zero routing errors. It has not replaced the
eight-port `gdn_gemv` in the full model.

Use [decode_disaggregated_gemv.md](decode_disaggregated_gemv.md) for the
measured evolution of the current decode path and [optimization_log.md](optimization_log.md)
for the chronological optimization record, including retired prefill designs.

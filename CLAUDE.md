# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Principle: Real-Data-Based Conclusions

**Every conclusion must be grounded in actual data observed from a runtime — a real build, synthesis, place-and-route, or on-card run — not inferred from previous logs, prior builds, or plausible reasoning.** When the available logs do not contain the specific fact needed, do not extrapolate from a similar earlier run: instrument and re-run to observe the real behavior directly. Concretely:

- **Reproduce and instrument** rather than infer. If a failure's root cause is not explicit in the log, re-run the exact failing step with added verbosity/introspection (e.g. a copied Vivado `link_design` wrapper with `-verbose` + `report_ip_status`) and read what actually happened. Triangulating from other builds is a hypothesis, not a conclusion.
- **Use the tool's own reports for design decisions.** Diagnose place-and-route/resource problems from `report_design_analysis -congestion`, `report_qor_suggestions`, and `report_utilization` on the actual (failed) checkpoint — not from an estimate or a past run. Let the measured numbers (per-SLR BRAM/URAM/LUT, congestion level, SLL crossing) pick the fix.
- **Change one variable, then measure.** Attribute an outcome only after a controlled run isolates that variable (e.g. a clean-cache relink to rule out stale state; a tool-version swap to confirm a tool bug).
- **State the evidence boundary.** Distinguish what a run *proves* from what it merely *suggests*, and say plainly when a fact is not obtainable from the current artifacts (so the answer is "not tool-exposed; needs an instrumented run," not a guess).

This is load-bearing: the 32-port `link_design` crash was pinned only by an instrumented rerun (a Vivado 2022.1 use-after-free, fixed in 2022.2), and the routing fix (recurrent state BRAM→URAM) came straight from `report_qor_suggestions`/`report_design_analysis` output — both would have been mis-diagnosed by inference from prior logs.

## Project Overview

**Hardware accelerator (Vitis HLS) for GatedDeltaNet inference**, based on the paper *Gated Delta Networks: Improving Mamba2 with Delta Rule* (ICLR '25, NVIDIA). The primary development target is `c_impl/`, which contains the HLS-synthesizable **C++** implementation of the GatedDeltaNet forward pass and the on-card hardware flow for the **Xilinx Alveo U55C**. The Python code (`lit_gpt/`, `pretrain.py`, `scripts/`) and the original Triton/FLA kernels serve as **golden references** for correctness verification.

Paper: https://arxiv.org/abs/2412.06464
Checkpoint: `m-a-p/1.3B-100B-GatedDeltaNet-pure` (HuggingFace) — consumed by `scripts/export_gdn_c.py` to produce the flat `.gdnw` weight blob.

For the **current decode-only accelerator**, start with `c_impl/doc/architecture.md`; use `c_impl/doc/decode_disaggregated_gemv.md` for the integrated GEMV evolution and measured results. Retired prefill designs remain only as historical entries in `c_impl/doc/optimization_log.md`.

## Model Configuration (GatedDeltaNet-1.3B)

Dimensions live in `c_impl/gdn_model.h` (the `GDN_*` constants) and `lit_gpt/config.py`.
Recurrent state per layer: 8 heads × 256 × 256 FP32 = 2 MB. Full weight blob ≈ 5.6 GB.

## Build & Run Commands

All accelerator work happens in `c_impl/`. The kernel is now **decode-only** — the GPU prefills the prompt and exports a fixed-size recurrent+conv state to disk, and the FPGA decodes token-by-token from it (see *Decode* below). The prefill GEMM, the prefill tops, and their testbenches/TCLs were removed. The Makefile (`make -C c_impl help`) drives three artifact classes: the host testbench (`gdn_eval`), the v++ kernel (`.xo`/`.xclbin`), and the XRT host program.

### Native C++ testbench (csim entry point — fast iteration & parity)
```bash
make -C c_impl                              # builds gdn_eval only (decode-only csim; c++ -O3 -std=c++14)
./c_impl/gdn_eval <weights.gdnw> <fixture.gdnreq> <out.json> --decode --decode-from-state <state.gdnstate> [--decode-len N]
```
`gdn_eval` is now a **decode-only** csim and hard-requires `--decode --decode-from-state`. The native build `#include`s `hls_stream.h`, so it needs the Vitis HLS include dir (`XILINX_HLS_INC`, default `/tools/Xilinx/Vitis_HLS/2022.1/include`). The old `gdn_attn_test` / `gdn_matmul_test` harnesses are retired — the decode-only pivot removed the prefill tops (`gdn_attn_forward`, `gdn_matmul_top`/`gdn_matmul2d_top`) they drove, so the default `host_tb` builds only `gdn_eval`.

### Parity testing (prefill era — vs Python golden)
```bash
cd c_impl && bash test_parity.sh            # rebuild, run every fixtures_smoke/*.gdnreq, diff vs results_smoke_python/
```
Tolerance 1e-3 (observed diffs ~1e-5). Diff logic is `scripts/check_gdn_c_parity.py`. **Note:** this is the prefill-parity flow; it no longer runs as-is on the decode-only branch (`gdn_eval` rejects non-decode invocations). The committed `results_smoke_python/` and the diff script remain; decode correctness is now gated by the check below.

### Decode correctness + TPOT (vs cached GPU golden — no GPU needed)
```bash
bash scripts/decode_correctness_check.sh            # full: 1 example × 32 steps, gates on exact match
bash scripts/decode_correctness_check.sh --fast     # 1×6 smoke (~1–2 min) — used by the inference-edit hook
./c_impl/gdn_eval <w.gdnw> fixtures_decode/decode.gdnreq <out.json> --decode --decode-from-state <state.gdnstate> [--decode-len N]
./c_impl/host.exe <xclbin> <w.gdnw> fixtures_decode/decode.gdnreq <out.json> 0 --decode --decode-from-state <state.gdnstate> [--decode-len N]   # on-card
```
The GPU prefills a prompt and exports the fixed-size recurrent+conv state with `scripts/export_gdn_state.py` → `.gdnstate` (~50 MB); the FPGA decodes from it through the `gdn_gemv` engine. The decode fixture (`fixtures_decode/decode.gdnreq`) and fp32 GPU golden (`results_decode_golden/`) are committed, but `decode_ex0.gdnstate` is **gitignored/regenerable** (like the `.gdnw` weight blob) — so the gate and the standing hook require both to be generated locally first (`export_gdn_state.py` + `export_gdn_c.py weights`). On-card decode is **bit-exact** to the golden over 64 tokens and **flat at 121.4 ms/token** (at 150 MHz) for a complete decode step (forward + lm_head + argmax all on-chip) — see `doc/decode_disaggregated_gemv.md`. A PostToolUse hook in `.claude/settings.json` auto-runs the fast check on every edit to `c_impl/gdn_model.{cpp,h}`, `host.cpp`, `gdn_eval.cpp`, `lit_gpt/gated_delta_net.py` — keep it passing.

### Vitis HLS synthesis (csim/csynth/cosim — no board needed)
```bash
cd c_impl
vitis_hls -f test.tcl                        # decode-only gdn_forward (24 layers, 1 token/step)
```
Targets Alveo U55C (`xcu55c-fsvh2892-2L-e`) at 100 MHz (10 ns clock). `test_single_GDN_attn_synth.tcl` (top `gdn_attn_forward`) and `test_matmul.tcl` (top `gdn_matmul_top`) are **stale** — both set tops the decode-only pivot removed. The TCL-variant map in `c_impl/doc/architecture.md` §2 predates that pivot.

### On-card hardware flow (v++ → bitstream → run on U55C)
```bash
cd c_impl
make run_hw                                  # xo → xclbin → host → ./host.exe on the board (dependency-driven)
make run_hw FIXTURE=fixtures_smoke/piqa.gdnreq HW_MAX_EX=4   # override input / cap examples
```
Phase times (cold): `xo` ~30–60 min, `xclbin` ~4–6 hr, `host` seconds. Individual phases: `make xo|xclbin|host`. Clean: `make clean|clean_hw|distclean`.

**Critical knob — kernel frequency** (`FREQ`, Makefile default 100): the committed decode artifact runs at **150 MHz**, built via over-synth (xo at `FREQ=250`, link at `FREQ=150`; routed WNS +0.025 ns, bit-exact — see `doc/decode_disaggregated_gemv.md` §6g/§6h). A plain `FREQ=100` build is functional but slower. The U55C gen3x16 platform defaults to 300 MHz, which the kernel cannot meet, so `--kernel_frequency` must override it (at 300 MHz the route fails, WNS ≈ −8.7 ns). The link is shaped by `hw.cfg` in `c_impl/` — kernel instance, the **disjoint HBM bank map** for the 8 GEMV weight readers + activations (overlapping `sp=` ranges → `std::bad_alloc`), and congestion-oriented Vivado strategies (`SSI_SpreadLogic_high`). `pblock_pe_split.tcl` floorplanned the prefill 16-PE systolic grid across SLR0/SLR1; it is **disabled** in the decode-only build (no systolic grid; `SSI_SpreadLogic_high` handles SLR spreading) but kept for reference.

### Exporting weights, state & fixtures (Python golden reference)
```bash
python scripts/export_gdn_c.py weights    --output c_impl/artifacts/gdn-1.3b-f32.gdnw
python scripts/export_gdn_state.py        ...   # GPU prefill → c_impl/fixtures_decode/*.gdnstate (recurrent+conv state)
python scripts/export_gdn_c.py fixtures   --tasks piqa hellaswag --output-dir c_impl/fixtures_smoke
```
`export_gdn_state.py` is the decode handoff producer (it self-checks bit-exactness vs the cache-decode golden). `export_block_fixture.py` (→ `.gdnblk`) targeted the retired single-layer attention harness.

### Running Python golden reference
```bash
python scripts/compare_gdn_c.py --fixture c_impl/fixtures_smoke/piqa.gdnreq --output results/piqa_python.json --device cuda --dtype float32
python scripts/check_gdn_c_parity.py --python-dir c_impl/results_smoke_python --c-dir c_impl/results_smoke_c --output c_impl/results_smoke_parity.json
python scripts/fla_lm_eval.py               # lm-eval-harness evaluation
```

## Architecture

### HLS Accelerator (`c_impl/`) — primary target
Per-file internals (kernel top, `gdn_gemv`, submodules, pragma conventions, `host.cpp`, formats)
live in `c_impl/CLAUDE.md`, which loads automatically when working under `c_impl/`.

### Golden Reference: Python Model (`lit_gpt/`)
Prefer the `gated_delta_rule_ops/fla_version/` kernels for golden runs — the HLS C++ mirrors them
function-for-function (mapping table in `README.md` / `c_impl/README.md`). `pretrain.py` and
`packed_dataset.py` are training-only and not exercised by the synthesis flow.

## Key Design Patterns & Current Focus

- The HLS C++ mirrors the Python computation graph exactly to maintain numerical parity (validated end-to-end within 1e-3, observed ~1e-5; on-card Wikitext perplexity matches Python golden to ~1e-7).
- **Optimization arc (prefill, on U55C):** the runtime was weight-HBM-traffic bound. Sequenced levers took wikitext-2048 prefill from **25.9 min → 4.2 min (~6.2×)**: weight-stationary blocking (kills ~95× weight re-reads), 512-bit bursted weight reads (aligned base + dedicated AXI bundle), Pack16 activation widening, and splitting activations across 3 HBM channels. Prefill is now **compute-bound** (matmul ≈ 76% of the kernel). PE-grid widening (Phase C) was attempted and **reverted** — it's a prefill lever with high routing risk and no decode benefit.
- **The accelerator is now decode-only (TPOT-focused).** Decode is a GEMV (1 token/step), **weight-bandwidth bound** not compute bound, so the prefill GEMM (16×16 systolic grid) was *removed* and replaced by the activation-stationary `gdn_gemv`. The disaggregated split (`doc/decode_disaggregated_gemv.md`): the GPU prefills + exports a fixed-size recurrent+conv state (`.gdnstate`, ~50 MB — free for a linear-attention model, no growing KV cache), and the FPGA decodes from it.
- **The decode work is built and on-card bit-exact.** Status: a complete decode step (forward + lm_head + argmax all on-chip) is **flat at 121.4 ms/token, bit-exact** to the GPU golden over 64 tokens — **3.9× over the 470 ms baseline**, via the multi-reader bandwidth ladder, a 150 MHz clock re-synth (§6g), and the gemv loop-flatten (§6h). The committed weight stream is sharded across **8 parallel HBM readers** (`GEMV_CHANNELS=8`), 3.34× over the single-port floor with diminishing returns per doubling. The 121.4 ms splits into **~68 ms gemv (~56%, weight-bandwidth bound) + ~53 ms serial floor (~44%)** — recurrent-attention state I/O on one HBM[0] master + the gated-delta-rule update (~27 ms) + conv/tiny, which more readers can't touch. The FPGA is now ~3.6× the A100's launch-bound 33.5 ms fp32, so the defensible angle is perf/Watt. Open levers: **more parallel readers** (`GEMV_CHANNELS=16` is under active investigation — it routes, but is blocked on SLR0-jam timing where the platform + 16 wide readers crowd the HBM south edge and scatter the distributed recurrent state; **uncommitted**, see `doc/decode_disaggregated_gemv.md`), overlap/parallelize the serial state I/O (attack the ~53 ms floor), INT8 weights (4× less traffic, costs bit-exactness). The fast correctness check runs as a standing hook on inference-code edits — keep it green.
- **The 32-port tiled GEMV microbenchmark is routed and verified on-card** (branch `decode_gemv_tiled`). The isolated mono-kernel in `c_impl/microbench/gemv_tile/` uses 32 512-bit HBM readers, eight four-port compute clusters distributed 2/3/3 across the U55C SLRs, and hierarchical SLR-local collectors. It routes with zero errors and passes synthetic plus real layer-0 `q_proj` parity. The requested 150 MHz implementation scales to 130.6 MHz because setup WNS is -0.985 ns, but sustains **263.063 GB/s and 131.531 GFLOP/s**, 98.353% of its clock-rate ceiling. Post-route analysis shows SLR1 at 90.47% CLB / 96.88% BRAM and SLR0-SLR1 SLL use at 95.03%; the next timing iteration should split four-port clusters rather than add replication to the congested route. See the microbenchmark README and `optimization_log.md`.
- For the decode GEMV the deep reference is `doc/decode_disaggregated_gemv.md`; retired prefill results remain in `doc/optimization_log.md` only.

## Dependencies

**HLS / hardware**: Vitis HLS + Vitis 2022.1, targeting Alveo **U55C** (`xcu55c-fsvh2892-2L-e`, platform `xilinx_u55c_gen3x16_xdma_3_202210_1`). On-card runs use XRT (default `/opt/xilinx/xrt`).

**Native C++ build**: any C++14 compiler + `libm`, plus the Vitis HLS include dir for `hls_stream.h`. No BLAS.

**Python (golden reference)**: container-based — see `Dockerfile` for the pinned versions.

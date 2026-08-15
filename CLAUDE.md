# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Principle: Real-Data-Based Conclusions

**Every conclusion must be grounded in actual data observed from a runtime — a real build, synthesis, place-and-route, or on-card run — not inferred from previous logs, prior builds, or plausible reasoning.** When the available logs do not contain the specific fact needed, do not extrapolate from a similar earlier run: instrument and re-run to observe the real behavior directly. Concretely:

- **Reproduce and instrument** rather than infer. If a failure's root cause is not explicit in the log, re-run the exact failing step with added verbosity/introspection (e.g. a copied Vivado `link_design` wrapper with `-verbose` + `report_ip_status`) and read what actually happened. Triangulating from other builds is a hypothesis, not a conclusion.
- **Use the tool's own reports for design decisions.** Diagnose place-and-route/resource problems from `report_design_analysis -congestion`, `report_qor_suggestions`, and `report_utilization` on the actual (failed) checkpoint — not from an estimate or a past run. Let the measured numbers (per-SLR BRAM/URAM/LUT, congestion level, SLL crossing) pick the fix.
- **Change one variable, then measure.** Attribute an outcome only after a controlled run isolates that variable (e.g. a clean-cache relink to rule out stale state; a tool-version swap to confirm a tool bug).
- **State the evidence boundary.** Distinguish what a run *proves* from what it merely *suggests*, and say plainly when a fact is not obtainable from the current artifacts (so the answer is "not tool-exposed; needs an instrumented run," not a guess).

This is load-bearing: the 32-port `link_design` crash was pinned only by an instrumented rerun (a Vivado 2022.1 use-after-free, fixed in 2022.2), and the routing fix (recurrent state BRAM→URAM) came straight from `report_qor_suggestions`/`report_design_analysis` output — both would have been mis-diagnosed by inference from prior logs.

## Iteration Record & Commit Discipline

`AGENTS.md` defines a second hard rule that governs this repo, restated here because it is easy to violate and expensive to undo:

1. **Every** iteration gets a `c_impl/doc/optimization_log.md` entry before the next one starts — including failures, stops, and rejects. Record the hypothesis, source/config/Tcl SHA-256s, the command and target frequency, what validation ran, HLS cycles/resources/II, congestion and timing, on-card results, and an explicit retained/rejected/inconclusive verdict. Never omit a negative result; name the exact stage and reason it failed. This log is why 30+ routing attempts are not being re-run blindly.
2. **Never commit a negative, neutral, inconclusive, or stopped iteration.** Keep its log entry in the working copy, revert the source/config that caused it, and continue from the last demonstrated improvement. Commits mark demonstrated improvements only.
3. Only after an iteration demonstrates a real improvement: update `doc/architecture.md` and the relevant block doc, then commit the source, build/config/Tcl, and all accumulated log entries in focused commits. Never commit build products, logs, weight blobs, or `.gdnstate`.
4. Always label evidence by strength — native-only, csynth, routed, timing-closed, or on-card — and never promote an intermediate result to production status.

**Consequence for the working tree:** an uncommitted diff to `gdn_model.cpp`/`hw*.cfg` is normal here — it is usually an in-flight experiment that has not yet earned a commit. Check `optimization_log.md` and `git diff` before assuming the working tree is the production design.

## Project Overview

**Hardware accelerator (Vitis HLS) for GatedDeltaNet inference**, based on the paper *Gated Delta Networks: Improving Mamba2 with Delta Rule* (ICLR '25, NVIDIA). The primary development target is `c_impl/`, which contains the HLS-synthesizable **C++** implementation of the GatedDeltaNet forward pass and the on-card hardware flow for the **Xilinx Alveo U55C**. The Python code (`lit_gpt/`, `pretrain.py`, `scripts/`) and the original Triton/FLA kernels serve as **golden references** for correctness verification.

Paper: https://arxiv.org/abs/2412.06464
Checkpoint: `m-a-p/1.3B-100B-GatedDeltaNet-pure` (HuggingFace) — consumed by `scripts/export_gdn_c.py` to produce the flat `.gdnw` weight blob.

### Which document to trust

`c_impl/doc/README.md` is the **index** and says which document is current versus historical. In short:

| Document | Status |
|---|---|
| `c_impl/doc/architecture.md` | **Authoritative** description of the production kernel (currently Iter57) — data flow, HBM map, per-layer schedule, physical design, measured result. Start here. |
| `c_impl/doc/optimization_log.md` | Exhaustive chronological record of *every* iteration, including failures. The §"Integrated 32-port `gdn_forward`" table is required reading before proposing any floorplan/config change — 30+ variants have already been built. |
| `c_impl/doc/decode_disaggregated_gemv.md` | How the decode design got here (single-reader → 8 → 32 ports). History, not the current spec. |
| `c_impl/doc/cycle_optimization_roadmap.md` | **Proposed** future stages. Targets, not implemented hardware. |
| `c_impl/microbench/gemv_tile/README.md` | The standalone 32-port GEMV microbenchmark (separate kernel from `gdn_forward`). |
| `c_impl/README.md` | **Stale** — still describes the retired C/prefill design (`gdn_model.c`, `gdn_attn_forward`, `test_single_GDN_attn.tcl`, gcc/C11). Do not cite it for current behavior. |
| `README.md` (root) | Mixed: its Python↔HLS function-mapping table is still valid; its performance and flow claims are prefill-era. |

Every status claim must say which thing it refers to: the production integrated kernel, a historical iteration, a roadmap target, or the microbenchmark.

## Model Configuration (GatedDeltaNet-1.3B)

Dimensions live in `c_impl/gdn_model.h` (the `GDN_*` constants) and `lit_gpt/config.py`.
Hidden 2048, 8 heads × 256 head-dim, MLP 5632, 24 layers, conv kernel 4, vocab 32000, **1 token per `gdn_forward` call**.
Recurrent state: 2 MB per layer (8 heads × 256 × 256 FP32), **48 MiB** across 24 layers; conv state ≈ 1.69 MiB. Both persist in HBM across kernel calls. Full weight blob ≈ 5.6 GB.

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
The GPU prefills a prompt and exports the fixed-size recurrent+conv state with `scripts/export_gdn_state.py` → `.gdnstate` (~50 MB); the FPGA decodes from it through the `gdn_gemv` engine. The decode fixture (`fixtures_decode/decode.gdnreq`) and fp32 GPU golden (`results_decode_golden/`) are committed, but `decode_ex0.gdnstate` is **gitignored/regenerable** (like the `.gdnw` weight blob) — so the gate and the standing hook require both to be generated locally first (`export_gdn_state.py` + `export_gdn_c.py weights`). The Iter57 image is **bit-exact** over 64 tokens and measures **42.023540 ms/token at a timing-closed 100 MHz** for a complete decode step (forward + lm_head + argmax all on chip) — see `doc/architecture.md`. The native gate also checks every pre-argmax logit against an independent scalar LM head. A PostToolUse hook in `.claude/settings.json` auto-runs the fast check on every edit to `c_impl/gdn_model.{cpp,h}`, `host.cpp`, `gdn_eval.cpp`, `lit_gpt/gated_delta_net.py` — keep it passing.

### Vitis HLS synthesis (csim/csynth/cosim — no board needed)
```bash
cd c_impl
vitis_hls -f test.tcl                        # decode-only gdn_forward (24 layers, 1 token/step)
```
Targets Alveo U55C (`xcu55c-fsvh2892-2L-e`). `test.tcl` sources `hls_gdn_forward.tcl`, which carries the interface settings the hardware build depends on (`config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512`, kernel profiling off, `config_rtl -reset control -register_reset_num 0` to keep reset out of the global routing). **The same pre-TCL is passed to `v++ -c` via `--hls.pre_tcl`, so csynth here and the hardware `.xo` see identical interface config** — keep them in sync. The retired prefill TCLs (`test_single_GDN_attn_synth.tcl`, `test_matmul.tcl`) were deleted along with their tops.

### On-card hardware flow (v++ → bitstream → run on U55C)
```bash
cd c_impl
make run_hw                                  # xo → xclbin → host → ./host.exe on the board (dependency-driven)
make run_hw JOBS=16 HW_DEVICE=1              # override build jobs / card index
```
Phase times: `xo` ~30–60 min, `host` seconds. The 32-port `xclbin` link is **long and highly variable — 7 to 32 hours measured** across iter32–iter37 (`diagnostics/*/build.manifest` start/exit timestamps; Iter36's 130 MHz link took 31.9 h, mostly post-route `AggressiveExplore` phys-opt). Budget accordingly. Individual phases: `make xo|xclbin|host`. Clean: `make clean|clean_hw|distclean`.

**`make run_hw` is the sole production build-and-run entry.** It resolves the relocatable physical configuration, builds HLS at 150 MHz, links the demonstrated 100 MHz image, then runs exact 8-token and 64-token on-card gates. Do not add iteration-specific Make targets or launcher scripts; preserve historical commands in `optimization_log.md` instead.

**Kernel frequency:** `HLS_FREQ` defaults to **150 MHz** and `LINK_FREQ`/`FREQ` to **100 MHz**. The U55C platform defaults to 300 MHz, which this kernel cannot meet, so never omit the link override. A requested frequency is not an achieved frequency: verify `DATA_CLK` in the XCLBIN and report the per-clock WNS. The fixed 250 MHz `dma_ip_axi_aclk_1` is a separate timing gate even when the scalable kernel clock closes.

**The link recipe (`hw_f150_physical_islands.cfg` plus its Tcl hooks) is where the physical design lives.** It carries:
- `nk=gdn_forward:1:gdn_forward_1` and the **one-bank-per-master** HBM map: `weight_data_mm0..mm31` → `HBM[0..31]`. The shell's 32-master limit forces `aux_weights` and `workspace` to share the `mem_weights_mm0` master on `HBM[0]`. Overlapping `sp=` *ranges* cause `xrt::bo` `std::bad_alloc` on U55C — keep bank assignments disjoint (`probe_alloc.cpp` diagnoses this).
- `prop=run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-directive Default}` — overrides the `sdx_optimization_effort_high` that `--optimize 2` injects, which got OOM-killed on this shared host.
- relocatable `@C_IMPL_DIR@` Tcl paths, the Iter57 island/collector pblocks, measured DMA/reset fanout repairs, a structural post-place gate, `SSI_SpreadSLLs`, `NoTimingRelaxation`, and pre/post-route `AggressiveExplore` physical optimization.

`pblock_pe_split.tcl` floorplanned the retired prefill systolic grid and is **disabled**; it is kept for reference only.

### Hardware iteration workflow

Modify the source, the `run_hw` defaults, or the current relocatable config/Tcl
in place. The build directory includes HLS/link frequencies and job count, so
Make cannot silently reuse an XO synthesized for another schedule. Record the
source/config/Tcl hashes and exact override command in `optimization_log.md`.
Only a demonstrated positive iteration becomes the next `run_hw` default;
rejected variants are logged and reverted rather than retained as new targets.

**Launch discipline for multi-hour builds** (also in `AGENTS.md`): detach the build so it survives a dropped session, record its PID/log/exit-marker, and *immediately* attach a watcher (`tail --pid=<pid> -F <build-dir>/.../impl_1/runme.log`) rather than checking back blind. Watch Vivado's detailed `runme.log`, not the quiet wrapper log. If the session dies, only the watcher stops — the build continues, and `diagnostics/*/build.exit` records the outcome.

Vivado `get_cells` glob gotcha, learned twice (iter8, iter37C): `*` **spans `/`**, so `NAME =~ */grp_foo_fu_*` matches the hierarchy root *and* all 200k descendants. Use `get_cells -hierarchical -regexp` with an anchored pattern like `^.*/grp_gdn_recurrent_attention_fu_[0-9]+$` and validate the match count against a checkpoint before launching.

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
function-for-function (mapping table in the root `README.md`). `pretrain.py` and
`packed_dataset.py` are training-only and not exercised by the synthesis flow.

## Key Design Patterns & Current Focus

- The HLS C++ mirrors the Python computation graph exactly to maintain numerical parity (validated end-to-end within 1e-3, observed ~1e-5; on-card Wikitext perplexity matches Python golden to ~1e-7).
- **Optimization arc (prefill, on U55C):** the runtime was weight-HBM-traffic bound. Sequenced levers took wikitext-2048 prefill from **25.9 min → 4.2 min (~6.2×)**: weight-stationary blocking (kills ~95× weight re-reads), 512-bit bursted weight reads (aligned base + dedicated AXI bundle), Pack16 activation widening, and splitting activations across 3 HBM channels. Prefill is now **compute-bound** (matmul ≈ 76% of the kernel). PE-grid widening (Phase C) was attempted and **reverted** — it's a prefill lever with high routing risk and no decode benefit.
- **The accelerator is now decode-only (TPOT-focused).** Decode is a GEMV (1 token/step), **weight-bandwidth bound** not compute bound, so the prefill GEMM (16×16 systolic grid) was *removed* and replaced by the activation-stationary `gdn_gemv`. The disaggregated split (`doc/decode_disaggregated_gemv.md`): the GPU prefills + exports a fixed-size recurrent+conv state (`.gdnstate`, ~50 MB — free for a linear-attention model, no growing KV cache), and the FPGA decodes from it.
- **Production status — Iter57, on-card, bit-exact and timing-closed.** The integrated 32-port kernel routes with zero failed/unrouted nets and zero node overlaps, closes the 100 MHz kernel and fixed 250 MHz DMA clocks at +0.060/+0.003 ns WNS, and produces an exact 64-token trajectory. It measures **42.023540 ms/token / 4.202354M cycles**, 2.48% faster than Iter39C and 2.889× faster than the eight-port reference. The design is `GEMV_CHANNELS=32` / `GEMV_CLUSTERS=16` / two channels per cluster, with a **4/6/6** SLR-local collector cut, BRAM MM2S decoupling, activation residency, four packed state ports, two concurrent 16-column recurrent islands, registered collector boundaries, depth-2048 state queues, and on-chip strict argmax.
- **The measured ladder** (each step is an on-card, exact result): 121.4 ms (8-port reference) → 98.66 (Iter32, activation resident) → 75.06 (Iter35, DMA fanout repair) → 59.58 (Iter36, head-local recurrence) → 51.45 (Iter37, four state ports) → 47.08 (Iter38, merged layouts/concurrent state) → 43.09 (Iter39C, head-streamed convolution) → **42.02** (Iter57, timing-friendly recurrent islands). Total **2.889×** over the eight-port reference.
- **Getting the integrated 32-port design to route was the hard part, and it is exhaustively documented.** 30+ build variants were attempted; `optimization_log.md` records each one's floorplan, stage reached, and measured failure. Two rules from that campaign: *judge a lever by its physical distribution, not its total resource saving* (iter16 won by moving FIFOs out of SLR0 CLB into BRAM), and *frequency is not a congestion lever* (iter20 dropped 150→130 MHz with no congestion relief). **Do not re-run a listed experiment without stating why the outcome would differ.**
- **Open levers.** Continue the cycle roadmap from the timing-closed Iter57 base: head-chunked output projection, then chunk-streamed GU/SwiGLU/MLP-down. Preserve the physical island and registered-boundary structure until a replacement completes exact on-card validation. Longer-term compression (BF16/INT8, sparsity) can lower the dense weight floor but changes the exact-FP32 contract.
- **The standalone 32-port GEMV microbenchmark** (`c_impl/microbench/gemv_tile/`) is a *separate kernel* used to characterize the HBM ceiling in isolation: 32 512-bit readers, eight four-port clusters at 2/3/3 across SLRs, SLR-local collectors. At 130.6 MHz it sustains **263.063 GB/s / 131.531 GFLOP/s** — 98.353% of its clock-rate ceiling — and passes synthetic plus real layer-0 `q_proj` parity. Its numbers are **not** `gdn_forward` numbers; never quote them as decode performance.
  ```bash
  make -C c_impl/microbench/gemv_tile csim_full          # native parity, no board
  make -C c_impl/microbench/gemv_tile xo_full xclbin_full JOBS=8 FREQ=150 READ_OUTSTANDING=16
  make -C c_impl/microbench/gemv_tile run_full RUN_FREQ=130.6 TIMED_REPS=9   # saturation bandwidth
  make -C c_impl/microbench/gemv_tile run_layer0 RUN_FREQ=130.6              # real q_proj parity
  make -C c_impl/microbench/gemv_tile analyze            # post-route SLR/SLL/congestion report
  ```
  Pass the **achieved** clock as `RUN_FREQ` — it only scales the reported efficiency, but a wrong value silently misreports the ceiling.
- The fast decode-correctness check runs as a standing hook on inference-code edits — keep it green.

## Dependencies

**HLS / hardware**: targets Alveo **U55C** (`xcu55c-fsvh2892-2L-e`, platform `xilinx_u55c_gen3x16_xdma_3_202210_1`). On-card runs use XRT (default `/opt/xilinx/xrt`).

**Two Vitis versions are in play — this matters.** The integrated 32-port hardware build **must** use **2022.2** (`/tools/Xilinx/Vitis/2022.2`): 2022.1's `link_design` has a use-after-free that crashes on 32 ports, confirmed by an instrumented rerun. The `c_impl/Makefile` uses Vitis 2022.2 for `v++`; `XILINX_HLS_INC` still defaults to the compatible 2022.1 header path for native compilation. The standalone microbenchmark remains documented against 2022.1.

**Native C++ build**: any C++14 compiler + `libm`, plus the Vitis HLS include dir for `hls_stream.h`. No BLAS.

**Python (golden reference)**: container-based — see `Dockerfile` for the pinned versions.

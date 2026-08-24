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
| `c_impl/doc/architecture.md` | **Authoritative** description of the production kernel (currently Iter61) — data flow, HBM map, per-layer schedule, physical design, measured result. Start here. |
| `c_impl/doc/optimization_log.md` | Exhaustive chronological record of *every* iteration, including failures. The §"Integrated 32-port `gdn_forward`" table is required reading before proposing any floorplan/config change — 30+ variants have already been built. |
| `c_impl/doc/decode_disaggregated_gemv.md` | How the decode design got here (single-reader → 8 → 32 ports). History, not the current spec. |
| `c_impl/doc/cycle_optimization_roadmap.md` | **Proposed** future stages. Targets, not implemented hardware. |
| `c_impl/doc/recurrent_attention.md`, `depthwise_conv.md`, `output_norm.md` | The active non-GEMV compute blocks. Current as *block* descriptions; their embedded synthesis tables are historical. |
| `c_impl/doc/decode_premise.md` | The GPU measurements that justify the decode-only partition (~35 ms/token, flat across context). Motivation, not a spec. Its "why GDN looks slow on GPU" section attributes the cost to launch overhead — that diagnosis is correct, but see the side-experiment caveat under *Key Design Patterns*: the overhead is also removable on the GPU. |
| `c_impl/doc/fp32_bf16_quality_evaluation.md` | **In progress** (see *Checkpoint quality evaluation* below) — a GPU-side precision study, not a statement about the kernel. |
| `c_impl/microbench/gemv_tile/README.md` | The standalone 32-port GEMV microbenchmark (separate kernel from `gdn_forward`). |
| `c_impl/README.md` | Current build and verification entry points for the decode-only accelerator. |
| `README.md` (root) | Current repository overview and links to the authoritative architecture documents. |

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
`gdn_eval` is a **decode-only** csim and hard-requires `--decode --decode-from-state`. The native build `#include`s `hls_stream.h`, so it needs the Vitis HLS include dir (`XILINX_HLS_INC`, default `/tools/Xilinx/Vitis_HLS/2022.1/include`). Retired prefill and standalone matmul/attention harnesses are available through Git history; the active native target is only `gdn_eval`.

### Decode correctness + TPOT (vs cached GPU golden — no GPU needed)
```bash
bash scripts/decode_correctness_check.sh            # full: 1 example × 32 steps, gates on exact match
bash scripts/decode_correctness_check.sh --fast     # 1×6 smoke (~1–2 min) — used by the inference-edit hook
./c_impl/gdn_eval <w.gdnw> fixtures_decode/decode.gdnreq <out.json> --decode --decode-from-state <state.gdnstate> [--decode-len N]
./c_impl/host.exe <xclbin> <w.gdnw> fixtures_decode/decode.gdnreq <out.json> 0 --decode --decode-from-state <state.gdnstate> [--decode-len N]   # on-card
```
The GPU prefills a prompt and exports the fixed-size recurrent+conv state with `scripts/export_gdn_state.py` → `.gdnstate` (~50 MB); the FPGA decodes from it through the `gdn_gemv` engine. The decode fixture (`fixtures_decode/decode.gdnreq`) and fp32 GPU golden (`results_decode_golden/`) are committed, but `decode_ex0.gdnstate` is **gitignored/regenerable** (like the `.gdnw` weight blob) — so the gate and the standing hook require both to be generated locally first (`export_gdn_state.py` + `export_gdn_c.py weights`). The Iter61 image is **bit-exact** over 64 tokens and measures **42.170227 ms/token at a timing-closed 100 MHz** for a complete decode step (forward + lm_head + argmax + full logit export, all on chip) — see `doc/architecture.md`. The native gate also checks every pre-argmax logit against an independent scalar LM head. A PostToolUse hook in `.claude/settings.json` auto-runs the fast check on every edit to `c_impl/gdn_model.{cpp,h}`, `host.cpp`, `gdn_eval.cpp`, `lit_gpt/gated_delta_net.py` — keep it passing.

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

**`make run_hw` is the production build-and-run recipe, and `bash run_hw_sbatch.sh` is how it reaches the cluster** (two chained Slurm jobs — see *Cluster environment* below; a card and 32 cores cannot be held by one job). It resolves the relocatable physical configuration, builds HLS at 150 MHz, links the demonstrated 100 MHz image, then runs exact 8-token and 64-token on-card gates. Do not add iteration-specific Make targets or launcher scripts; preserve historical commands in `optimization_log.md` instead.

**Kernel frequency:** `HLS_FREQ` defaults to **150 MHz** and `LINK_FREQ`/`FREQ` to **100 MHz**. The U55C platform defaults to 300 MHz, which this kernel cannot meet, so never omit the link override. A requested frequency is not an achieved frequency: verify `DATA_CLK` in the XCLBIN and report the per-clock WNS. The fixed 250 MHz `dma_ip_axi_aclk_1` is a separate timing gate even when the scalable kernel clock closes.

**The link recipe (`hw_f150_physical_islands.cfg` plus its Tcl hooks) is where the physical design lives.** It carries:
- `nk=gdn_forward:1:gdn_forward_1` and the **one-bank-per-master** HBM map: `weight_data_mm0..mm31` → `HBM[0..31]`. The shell's 32-master limit forces `aux_weights` and `workspace` to share the `mem_weights_mm0` master on `HBM[0]`. Overlapping `sp=` *ranges* cause `xrt::bo` `std::bad_alloc` on U55C — keep bank assignments disjoint (`probe_alloc.cpp` diagnoses this).
- `prop=run.__KERNEL__.{STEPS.SYNTH_DESIGN.ARGS.MORE OPTIONS}={-directive Default}` — overrides the `sdx_optimization_effort_high` that `--optimize 2` injects, which got OOM-killed on this shared host.
- relocatable `@C_IMPL_DIR@` Tcl paths, the Iter61 island/collector pblocks, measured DMA/reset fanout repairs, a structural post-place gate, `SSI_SpreadSLLs`, `NoTimingRelaxation`, and pre/post-route `AggressiveExplore` physical optimization.

`pblock_pe_split.tcl` floorplanned the retired prefill systolic grid and is **disabled**; it is kept for reference only.

### Hardware iteration workflow

Modify the source, the `run_hw` defaults, or the current relocatable config/Tcl
in place. The build directory includes HLS/link frequencies and job count, so
Make cannot silently reuse an XO synthesized for another schedule. Record the
source/config/Tcl hashes and exact override command in `optimization_log.md`.
Only a demonstrated positive iteration becomes the next `run_hw` default;
rejected variants are logged and reverted rather than retained as new targets.

**Launch discipline for multi-hour builds** (also in `AGENTS.md`): detach the build so it survives a dropped session, record its PID/log/exit-marker, and *immediately* attach a watcher (`tail --pid=<pid> -F <build-dir>/.../impl_1/runme.log`) rather than checking back blind. Watch Vivado's detailed `runme.log`, not the quiet wrapper log. If the session dies, only the watcher stops — the build continues, and `diagnostics/*/build.exit` records the outcome.

**All hardware work goes through Slurm on the `accl` cluster.** Never run a
build, `xbutil`, or an on-card test outside a job, and never `ssh` to a node to
reach a card — the device nodes are mode `0666`, so an SSH shell sits in no
cgroup and can open a card allocated to someone else. The `accl-cluster` skill
carries the full cluster contract; the rules that bite this project:

- **Build and on-card run are two jobs, always.** `build` grants no FPGA and
  `light` grants only 8 cores, so no single job can link the image and then run
  it. `bash c_impl/run_hw_sbatch.sh [TAG]` submits both and chains the on-card
  job with `afterok`, so it is cancelled automatically if the build fails.
  `slurm/hw_build.slurm` and `slurm/hw_oncard.slurm` are the two halves.
  **`make run_hw` is still the build-and-run recipe, but it can only be invoked
  whole outside Slurm** — under Slurm the on-card job calls it with `-o` on the
  image so make runs the gates and can never start a link in an 8-core job.
- **Set `--time` on any link.** The `build` default is 12 h and links here have
  taken 7–32 h; `MaxTime` is 3 days on every partition, so a long link must ask
  for what it needs and can never exceed `3-00:00:00`.
- **Never pass `--qos`.** The partition selects it. A real-but-mismatched QOS is
  accepted and then pends forever.
- **One FPGA and one GPU per user across all jobs**, even though `light` now
  allows four concurrent jobs. An idle allocation still holds its card, so a
  forgotten `salloc --no-shell` blocks every later on-card test while other
  cards sit free. Check with `squeue -u $USER -o '%i %j %b %T %M'` before
  blaming the queue.
- **Memory.** Vivado peaks at **~51 GB** on this design (Iter57 peaked 34.7 GB;
  Iter61 reached 51 GB). `light` caps a job at 32 GB and the OOM killer took
  Vivado there at 27 GB, during *Design Initialization*, before placement ran.
  `build` allows 196800M; the scripts ask for 128 GB. `light` and `vnc` (8 GB)
  cannot build this design at all.
- **Not every node can link for the U55C.** Probed 2026-08-22: the platform is
  present on `acclnode01`, `acclnode03` and `harrier`, and **absent on
  `acclnode04` and `acclnode05`** (`acclnode04` has no XRT either). The build
  script excludes those two. Re-probe after any cluster change rather than
  trusting this list — pinning to `harrier` also works but needlessly
  serialises against card jobs.
- **Device index.** Inside a `--gres=fpga:u55c:1` job, `ConstrainDevices=yes`
  hides every card the job does not own, so the XRT index is **0** even on a
  host with three cards. Do not carry over an index measured outside Slurm (the
  old "U55C is index 2" note applied to a bare SSH shell). The on-card script
  asserts exactly one visible card and that it is a U55C, then uses 0; a wrong
  index fails with a bare `err = -22`.
- **The filesystem trap is gone.** `/home/yaoz0b` is now the same NFS share
  (`10.0.0.11:/mnt/homes`) on the login node and on every compute node —
  verified by tailing, from `acclhead1`, a log being written by a job on
  `acclnode01`. The old "acclnode03/04/05 cannot write to /home" note, and the
  `--nodelist=harrier` pin it forced, no longer apply. Editing on `acclhead1`
  and submitting is safe.
- **Vivado on NFS is slower than on node-local NVMe** (`/tmp/$USER-$SLURM_JOB_ID`).
  The current scripts still link in the repo directory, because the relocatable
  `@C_IMPL_DIR@` substitution and the Tcl hooks are proven there and moving a
  7–32 h link is not a change to make casually. Unmeasured on this design;
  treat as an open lever, not a rule.

**Two stale-artifact traps in reused build directories.** The build directory name embeds the job count, so `make xo JOBS=8` builds into `.o8` while `.o16` keeps an older report — comparing the wrong `csynth.rpt` produces a completely wrong resource delta. And `impl_1/runme.log` survives from the previous run, so a congestion table read early in a new build is the *previous* build's. Check the file's mtime against the job start before trusting either.

Vivado `get_cells` glob gotcha, learned twice (iter8, iter37C): `*` **spans `/`**, so `NAME =~ */grp_foo_fu_*` matches the hierarchy root *and* all 200k descendants. Use `get_cells -hierarchical -regexp` with an anchored pattern like `^.*/grp_gdn_recurrent_attention_fu_[0-9]+$` and validate the match count against a checkpoint before launching.

### Exporting weights, state & fixtures (Python golden reference)
```bash
python scripts/export_gdn_c.py weights    --output c_impl/artifacts/gdn-1.3b-f32.gdnw
python scripts/export_gdn_c.py decode     --output-dir c_impl/fixtures_decode
python scripts/export_gdn_state.py        ...   # GPU prefill → c_impl/fixtures_decode/*.gdnstate (recurrent+conv state)
```
`export_gdn_state.py` is the decode handoff producer (it self-checks
bit-exactness versus the cached decode golden).

### Running Python golden reference
```bash
python scripts/compare_gdn_c.py --decode-golden c_impl/fixtures_decode/decode.gdnreq --output c_impl/results_decode_golden/decode.decode.json
python scripts/check_gdn_c_parity.py --decode --golden c_impl/results_decode_golden/decode.decode.json --c <candidate.json>
python scripts/fla_lm_eval.py               # lm-eval-harness evaluation
```

### Checkpoint quality evaluation (GPU-only — FP32 vs BF16, paper Tables 2/3/5)

A **separate workstream from the kernel**, on the `bf16` branch and now committed. It runs on an A100, touches no HLS
code, and answers two questions before BF16 is considered as a hardware lever: (1) does the
`m-a-p/1.3B-100B-GatedDeltaNet-pure` checkpoint reproduce the paper's Tables 2/3/5, and (2) does a
BF16-weight/BF16-activation cast retain that quality while the recurrent kernel still accumulates
state in FP32?

```bash
bash scripts/run_gdn_table3_eval.sh                     # Table 3: lm-eval short-context + commonsense
bash scripts/run_gdn_table2_eval.sh                     # Table 2: RULER S-NIAH (custom tasks)
python scripts/run_gdn_longbench_eval.py --model <ckpt-dir> --dtype float32|bfloat16 --output-dir <dir>
python scripts/convert_gdn_checkpoint_bf16.py <src> <dst>    # cast every float safetensors tensor to BF16
python scripts/summarize_gdn_quality_eval.py <root>          # root must contain fp32/ and bf16/
```

- **`DTYPE` defaults differ between the two shell launchers** — `run_gdn_table3_eval.sh` defaults to
  `float16`, `run_gdn_table2_eval.sh` to `float32`. A precision comparison must set `DTYPE`
  explicitly; taking the default silently evaluates a third precision. Both launchers are
  env-var-driven (`MODEL_ID`, `DTYPE`, `BATCH_SIZE`, `TASKS`, `OUTPUT_DIR`, `GDN_ATTN_MODE`, …) and
  forward extra argv to `fla_lm_eval.py`.
- Evaluation goes through the **`gdn_hf` lm-eval model type** registered in `scripts/fla_lm_eval.py`
  (an `HFLM` subclass that forces `config.attn_mode`, default `fused_recurrent`) — not stock `hf`.
- Task definitions live in the repo: `scripts/eval_tasks/gdn_ruler_table2` (passed via
  `--include_path`) and `scripts/eval_configs/longbench_v1_table5.json`.
- **Artifacts are deliberately outside the Git tree** at `/home/yaoz0b/gdn_precision_eval_20260817/{fp32,bf16}`.
  Converted checkpoints, lm-eval result JSON, and logs are never committed.
- Per-arm runner and checks: `scripts/run_gdn_precision_eval_arm.sh` runs one complete arm (Tables 2/3/5) for a given `MODEL_ID`/`DTYPE` — the three recorded arms differ only in those two values; `scripts/verify_bf16_conversion.py` validates a conversion before spending GPU hours; `scripts/analyze_gdn_eval_artifacts.py` provides the two analyses the summarizer does not (first-line Table 5 rescoring, paired per-sample flip counts).
- Acceptance thresholds were **fixed before results were inspected** and are recorded in the doc —
  treat them as pre-registered and do not retune them to fit an outcome.
- Status as of 2026-08-20: **complete** — FP32 and both BF16 arms have run all three tables at full sample counts, plus a BF16-recurrent-state follow-up. Earlier status as of 2026-08-17 was: **Table 3 FP32 complete and passing** (accuracy average 58.09 vs the
  paper's 55.32); the BF16 column and all of Tables 2/5 are still pending. Do not describe the BF16
  path as validated until those cells are filled.

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
- **Production status — Iter61, on-card, bit-exact and timing-closed.** The integrated 32-port kernel routes with zero failed nets and zero node overlaps, closes timing at **+0.003 ns WNS**, and produces an exact 64-token trajectory. It measures **42.170227 ms/token** at a timing-closed 100 MHz. Iter61 is Iter57 plus one thing: the LM head now streams its full 32,000-value logit vector to the workspace, costing **+0.35%** over Iter57's 42.023540. The design is `GEMV_CHANNELS=32` / `GEMV_CLUSTERS=16` / two channels per cluster, with a **4/6/6** SLR-local collector cut, BRAM MM2S decoupling, activation residency, four packed state ports, two concurrent 16-column recurrent islands, registered collector boundaries, depth-2048 state queues, on-chip strict argmax, and the logit FIFO.
- **The Iter61 lesson, which generalizes: put the FIFO between the block and the memory port, not the port inside the block.** `gemv32_store` is distributed across all three SLRs by the island pblocks, so an AXI write there is an AXI write everywhere. Iter58 routed logits through `gdn_gemv`'s shared `out` pointer — one interface across every GEMV call — and HLS sized that buffer for the vocabulary: **+1291 BRAM**, place_design DRC failed. Iter59 then tried to buy the BRAM back by moving FIFOs to URAM and made congestion **worse in all four directions**; `route_design` refused at level 7. Iter61 has `gemv32_store` fill an `hls::stream` that the *top level* drains, adding no port to the GEMV region at all: BRAM unchanged at 1728, URAM +8 of 912 free, LUT +0.4%, cycles +0.003%. **Judge a lever by where it puts its interfaces, not by its resource total.**
- **GPU reference point: ~35 ms/token** on an A100 80GB at batch 1, bf16, from `scripts/bench_tpot.py` — see `doc/decode_premise.md`. That is stock PyTorch + fla, the configuration a user actually gets, and it is the baseline the decode-only partition is argued against. It is flat across a 512x context range, which is the architectural claim that matters.
- **Caveat on that number, from a side experiment — not a new baseline.** Profiling shows the 35 ms is dominated by dispatch, not arithmetic: **1,981 kernel launches per token**, 58% of wall time in gaps, 4.1% of peak bandwidth. Capturing the per-token step as one CUDA graph reached **4.20 ms/token**, verified token-identical to eager (24/24 tokens, zero logit difference); `torch.compile` was slower than eager because dynamo recompiles per layer on fla's cache indexing. This is an **experiment on branch `worktree-gpu-decode-opt`** (`scripts/gpu_decode_{profile,optimize,verify}.py`), not merged and not the reference. Its bearing on this project: a hand-optimised GPU can go far below 35 ms, so do not treat the accelerator's margin against the stock number as the whole story, and be careful claiming raw-speed superiority. The durable claims are flat O(1) latency, constant memory with no KV cache, and performance per watt and per dollar.
- **The measured ladder** (each step is an on-card, exact result): 121.4 ms (8-port reference) → 98.66 (Iter32, activation resident) → 75.06 (Iter35, DMA fanout repair) → 59.58 (Iter36, head-local recurrence) → 51.45 (Iter37, four state ports) → 47.08 (Iter38, merged layouts/concurrent state) → 43.09 (Iter39C, head-streamed convolution) → 42.02 (Iter57, timing-friendly recurrent islands) → **42.17** (Iter61, +0.35% for on-chip logit export). Total **2.879×** over the eight-port reference.
- **Getting the integrated 32-port design to route was the hard part, and it is exhaustively documented.** 30+ build variants were attempted; `optimization_log.md` records each one's floorplan, stage reached, and measured failure. Two rules from that campaign: *judge a lever by its physical distribution, not its total resource saving* (iter16 won by moving FIFOs out of SLR0 CLB into BRAM), and *frequency is not a congestion lever* (iter20 dropped 150→130 MHz with no congestion relief). **Do not re-run a listed experiment without stating why the outcome would differ.**
- **Open levers, in order of measured promise.** (1) **Sub-byte weight compression** (INT4/INT3, non-power-of-two groups with dequantisation in the datapath) is the only lever with the right magnitude: it attacks the 4.2× bandwidth deficit directly, is something tensor cores cannot natively execute, and would take the per-token read from 5.195 GB toward 0.65 GB. (2) The cycle roadmap's remaining stages — head-chunked output projection, then chunk-streamed GU/SwiGLU/MLP-down — are percent-level; Iter57 gained 2.48% over Iter39C. (3) **BF16 weights** would roughly halve HBM bytes and land near 13–21 ms; note a hand-optimised GPU reaches 4.20 ms with *no* quantisation applied, so BF16 alone buys little differentiation. All of these retire the bit-exact decode gate and need the quality work in `fp32_bf16_quality_evaluation.md` repeated for the new format. Preserve the physical island and registered-boundary structure until a replacement completes exact on-card validation.
- **On-chip recurrent state is quality-safe but physically blocked.** BF16 state costs **0.02 accuracy points** on Table 3 (measured, `fp32_bf16_quality_evaluation.md`) and would halve the state to 24 MiB, inside the device's 33.8 MiB of URAM. But that is **683 URAM of 960 with only 320 per SLR**, while the recurrent islands are pblock-pinned to SLR2 — so the state cannot sit in the SLR that consumes it. Iter59 measured a weaker version of exactly this and congestion got worse in all four directions. The prize is also bounded: state traffic is 1.9% of per-token bytes.
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

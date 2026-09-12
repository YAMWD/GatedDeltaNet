# GatedDeltaNet decode accelerator

`c_impl/` contains the decode-only Vitis HLS implementation of
GatedDeltaNet-1.3B for the Alveo U55C (`xcu55c-fsvh2892-2L-e`). The model
shape is fixed at 24 layers, hidden size 2048, eight heads, head dimension
256, intermediate size 5632, convolution width four, and vocabulary size
32,000.

The current architecture and measured results are documented under `doc/`.
In particular:

- `doc/architecture.md` describes the top-level datapath and memory contract.
- `doc/decode_disaggregated_gemv.md` describes the 32-port GEMV engine.
- `doc/recurrent_attention.md` describes recurrent-state ownership and flow.
- `doc/cycle_optimization_roadmap.md` records the remaining optimization plan.
- `doc/optimization_log.md` is the complete positive and negative build log.

Historical source and build recipes are available through Git history rather
than being retained as inactive files in the production directory.

## Building the hardware from a clean clone

Four inputs are gitignored because they are large and regenerable. Produce
them first; everything else in the chain is committed.

```bash
python scripts/export_gdn_c.py weights            # 5.87 GB BF16-exact .gdnw
sbatch scripts/export_all_bf16_reference.slurm    # GPU logits + .gdnstate handoff
sbatch scripts/run_all_bf16_native_reference.slurm  # optional: native diagnostic
cd c_impl && bash run_hw_sbatch.sh <tag>          # build + on-card, two chained jobs
```

`GPU_LOGITS_REFERENCE` is a required on-card gate input; `LOGITS_REFERENCE`
is not, and defaults to empty. The root `CLAUDE.md` carries the full
committed-file manifest under *Reproducing a hardware build from a clean
clone*.

## Active source and build files

| File | Purpose |
|---|---|
| `gdn_model.cpp`, `gdn_model.h` | Synthesizable kernel and native support code |
| `gdn_eval.cpp` | Decode-only native correctness driver |
| `host.cpp` | XRT host and on-card full-logit validation |
| `Makefile` | Native, XO, XCLBIN, host, and complete `run_hw` flow |
| `test.tcl` | Integrated HLS synthesis entry point |
| `hls_gdn_forward.tcl` | HLS interface and RTL configuration |
| `hw_f150.cfg` | Production 150 MHz connectivity and physical hooks (`finish_f150_timing.tcl`, `check_iter75d_final_timing.tcl`, `apply_iter75d_control_fanout.tcl`, `apply_iter69_kernel_clock_f150.tcl`) |
| `reconcile_exact_clock.py` | Fail-closed DATA_CLK metadata reconciliation after the link |
| `run_hw_sbatch.sh` | Slurm build/test submission wrapper |
| `slurm/` | Build and U55C test job definitions |

## Native verification

Build the decode driver:

```bash
make -C c_impl
```

Run the short exact decode gate, or omit `--fast` for the full gate:

```bash
bash scripts/decode_correctness_check.sh --fast
```

The gate requires locally generated weight, state, and logit-reference files.
These large artifacts are intentionally not tracked by Git.

## HLS synthesis

With Vitis HLS 2022.2 configured:

```bash
cd c_impl
vitis_hls -f test.tcl
```

`test.tcl` synthesizes only `gdn_forward`; `gdn_eval.cpp` is registered solely
as a testbench. Production-specific cosim drivers have their own generated Tcl
files and fixtures.

## Hardware build and on-card validation

Submit hardware work through Slurm:

```bash
cd c_impl
bash run_hw_sbatch.sh
```

`make run_hw` on the login node runs `run_hw_sbatch.sh`, which freezes a
source snapshot and submits the build and FPGA test as separate jobs, chaining
the test with `afterok`. The card job runs the submission's frozen
`reproduction.Makefile` with `GDN_ONCARD=1`, so it can never start a link. The
demonstrated launch is `BUILD_EXCLUSIVE=user BUILD_NODE=acclnode01
BUILD_EXCLUDE=acclnode04,acclnode05,harrier make run_hw`; see
`doc/reproduce_f150.md`.

Use `make -C c_impl help` to display the configurable weights, state, fixture,
reference, frequency, device, and output paths.

## Generated files

Do not commit weight or state blobs, logit dumps, binaries, XOs, XCLBINs,
Vivado/Vitis work directories, implementation reports, or diagnostics. They
are reproducible from the source, configuration, and recorded commands.

# Repository Guidelines

## Project Structure & Module Organization

`c_impl/` is the primary target: a decode-only Vitis HLS C++ accelerator. Core compute lives in `gdn_model.cpp`/`.h`; `gdn_eval.cpp` is the native testbench, `host.cpp` is the XRT host, and `doc/decode_*.md` documents the current design. `lit_gpt/` is the PyTorch/Triton golden reference. `scripts/` exports weights, fixtures, and recurrent state and performs parity checks. `pretrain.py` is optional training infrastructure. Committed fixtures and expected JSON results live under `c_impl/fixtures_*` and `c_impl/results_*`. For decode work, prefer `c_impl/doc/decode_disaggregated_gemv.md`; parts of the READMEs describe the retired prefill flow.

## Build, Test, and Development Commands

- `make -C c_impl help` lists targets and configurable platform, frequency, and device knobs.
- `make -C c_impl` builds the native `gdn_eval` testbench with C++14. It requires Vitis HLS 2022.1 headers (`XILINX_HLS_INC`).
- `bash scripts/decode_correctness_check.sh --fast` runs the short exact-match decode gate; omit `--fast` for the full 32-step check.
- `cd c_impl && vitis_hls -f test.tcl` runs decode-kernel csim/csynth/cosim.
- `make -C c_impl run_hw` builds and runs the U55C hardware flow. Bitstream linking can take several hours and requires Vitis/XRT plus a board.
- For long-running builds, detach the build with a persistent PID, log,
  exit-code marker, and artifact paths so it survives chat interruption. If a
  build or validation gate is expected to need more than 10 minutes to finish,
  keep a bounded launch-sentry window for roughly two minutes immediately after
  submission. During that window, inspect the shared live log and scheduler
  state for fast wrapper, environment, compilation, csim, or dependency
  failures and fix/resubmit them promptly when they are unambiguous. Once the
  job has entered a genuinely long synthesis, placement, or routing stage,
  do not keep the current turn open by polling, waiting, or repeatedly reading
  its status. End the current chat turn after reporting the current stage, a
  best-effort ETA, and the absolute path to the active detailed log (prefer
  Vivado `impl_1/runme.log` over a quiet wrapper log). On a later explicit
  status request, inspect the process and log once, report the result, and end
  the turn again if more than 10 minutes are still expected. This rule avoids
  spending tokens on status polling and meaningless waiting.
- For every new Slurm job, put both `#SBATCH --output` and the detailed
  tool-specific live log under the shared repository diagnostics directory in
  `/home/yaoz0b/GatedDeltaNet/c_impl/diagnostics/<tag>/`. The login and compute
  nodes share `/home/yaoz0b`; do not put the only copy of a live log under a
  compute node's `/tmp`. Work products may remain in node-local `/tmp` for
  performance, but stdout/stderr must be visible from `acclhead1` in real time.
  Before ending the launch turn, verify the shared log exists and give the user
  a clickable absolute link to it, together with the Slurm job ID and ETA.

### Slurm-only hardware workflow

`accl-cluster.md` and the repository-root `SKILL.md` are the authoritative
cluster references. All Vitis hardware builds and all FPGA tests must run as
Slurm jobs; never build on `acclhead1`, and never SSH directly to an
accelerator node to use a card.

These submission rules apply prospectively to new jobs. Do not cancel,
requeue, migrate, or reconfigure an already pending or running job merely
because it predates the current policy; let it finish unless the user
explicitly asks otherwise.

- Run scheduler commands on `acclhead1` with
  `/opt/slurm/current/bin` on `PATH`. If the current host is already
  `acclhead1`, run them directly; otherwise use SSH. Do not accidentally query
  the unrelated `solar` cluster from another host.
- Never pass `--qos`. The partition selects the policy automatically.
- `/home/yaoz0b` is the same NFS export
  (`10.0.0.11:/mnt/homes/yaoz0b`) on `acclhead1` and the compute nodes. Source,
  Slurm output, live logs, exit markers, and final reports can therefore be
  shared directly. Continue staging heavy Vitis/Vivado working trees under
  `/tmp/$USER-$SLURM_JOB_ID` because NFS builds are slower.
- Submit compute-only HLS, RTL cosim, synthesis, and linking to `build`, request
  no accelerator GRES, and stay within 48 CPUs and 192 GiB. **Do not pin a
  build node by default.** Request the required Vitis/Vivado feature (for
  example `vivado2024.2`) and let Slurm select any eligible node; use
  `--nodelist` only when the user requests a particular host or a measured
  node-specific requirement leaves no alternative. Do not force these jobs
  onto `harrier`. Verify the required toolchain and U55C platform after
  allocation and before a production link. As of 2026-08-24, `acclnode01` and
  `acclnode03` advertise Vitis/Vivado 2022.2; `acclnode02` is not registered.
  A measured missing dependency may be handled with `--exclude` without
  pinning a replacement node. As of 2026-08-28, `acclnode05` advertises
  `vivado2024.2` but lacks the U55C platform, so exclude it from U55C builds.
- Submit hardware execution as a separate `light` job with
  `--gres=fpga:u55c:1`, at most 8 CPUs and 32 GiB. Do not assume the request
  routes to `harrier`: inspect the allocated node and XRT version, then confirm
  the expected U55C with `xbutil examine`. Make it `afterok`-dependent on the
  build job when both are submitted together.
- A card is usable only inside the job that requested it. Confirm
  `xbutil examine` shows exactly the allocated card before loading the XCLBIN.
  Do not infer allocation from `/dev/dri`.
- Source `/tools/Xilinx/Vitis/2024.2/settings64.sh` for builds (the version is
  pinned once as `VITIS_VERSION` in `c_impl/Makefile` — follow that knob if it
  moves) and `/opt/xilinx/xrt/setup.sh` for card runs. Keep the U55C platform
  at `xilinx_u55c_gen3x16_xdma_3_202210_1`.
- Treat `QOSMaxJobsPerUserLimit` as normal queueing when one job of that class
  already exists. For failures, start with `scontrol show job` and `sacct`, then
  inspect the Slurm output and detailed Vitis/Vivado log.
- Use persistent Slurm output, detailed live logs, status, and exit-code files
  in the shared workspace. Do not add a separate polling/mirroring supervisor
  when direct NFS logging is available. The existing greater-than-10-minute
  no-polling rule applies: hand back the job IDs, ETA, and clickable live-log
  links, then end the turn.

## Coding Style & Naming Conventions

Use four-space indentation and follow neighboring code; no repository-wide formatter is configured. Python uses `snake_case`, type hints, and `pathlib` where practical. C++ uses `snake_case` functions, `GDN*` types, and uppercase `GDN_*` constants. Keep C++14 compatibility and the Makefile warning flags clean. Preserve labeled HLS loops, trip-count pragmas, memory bundles, and floating-point operation order unless the associated parity and synthesis impact is measured.

## Testing Guidelines

There is no coverage threshold or general pytest suite. Accelerator changes should build natively and pass the fast decode gate; run the full gate before review. The check requires locally generated `.gdnw` weights and `.gdnstate` state files. Name new C++ harnesses `*_test.cpp` and workflow checks `*_check.sh`; keep fixture/result schemas backward compatible.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative, scoped subjects such as `decode: flatten GEMV MAC loop` and `doc: update timing results`. Keep each commit focused. PRs should explain the affected path, link relevant issues, list exact verification commands, and report correctness plus any latency, II, WNS, frequency, or resource changes. Note required hardware and attach concise report excerpts for QoR-sensitive changes.

### Iteration Record and Commit Discipline

Treat every named optimization or hardware-build iteration as a result that
must be preserved, whether it succeeds, fails, is stopped, or is rejected
after synthesis. Before starting the next iteration:

1. Append an entry to `c_impl/doc/optimization_log.md`. Record the hypothesis,
   source/config/Tcl identity, commands and target frequency, validation that
   ran, HLS cycles/resources/II when available, implementation congestion and
   timing when available, on-card correctness/performance when available, and
   the final retained/rejected/inconclusive verdict. Never omit a negative
   result; state the exact stage and reason it failed.
2. Do not create a commit for a negative, neutral, inconclusive, stopped, or
   rejected iteration. Record its result in the working copy of the optimization
   log, revert or exclude the implementation/configuration changes that caused
   it, and continue from the last demonstrated improvement. Never commit source,
   config, Tcl, launcher, or other changes that fail to improve the objective.
   Keep accumulated negative/neutral log entries uncommitted until a later
   positive iteration is ready to commit.
3. Only when an iteration demonstrates a real improvement, update the
   corresponding files in `c_impl/doc/`—especially `architecture.md` and the
   relevant block document—
   so they describe the new architecture, measured result, and reproduction
   command. Clearly distinguish native-only, csynth, routed, timing-closed, and
   on-card evidence; do not promote an intermediate result to production
   status. Every positive iteration must also update
   `c_impl/doc/cycle_optimization_roadmap.md`: mark the completed stage, replace
   estimates with measured evidence, rebase the current cycle reference when
   applicable, and revise the remaining targets or dependencies.
4. After that improvement is demonstrated and documented, commit the retained
   source architecture changes, necessary build/config/Tcl or launcher files,
   the positive result, and all accumulated optimization-log entries in focused
   commits. Include the roadmap update in the same positive commit series so
   the roadmap is never stale after a positive commit. Do not commit generated
   build products or logs.

Do not move on to the next optimization iteration with an unrecorded result.
Never commit an architectural or build change whose measured result is negative
or neutral; commits mark demonstrated improvements only.

## Generated Artifacts

Do not commit weight blobs, exported state, binaries, logs, or Vitis/Vivado work directories such as `c_impl/build.*`, `c_impl/_x*`, and `.Xil/`; these are ignored and regenerable.

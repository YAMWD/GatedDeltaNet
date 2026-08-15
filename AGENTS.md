# Repository Guidelines

## Project Structure & Module Organization

`c_impl/` is the primary target: a decode-only Vitis HLS C++ accelerator. Core compute lives in `gdn_model.cpp`/`.h`; `gdn_eval.cpp` is the native testbench, `host.cpp` is the XRT host, and `doc/decode_*.md` documents the current design. `lit_gpt/` is the PyTorch/Triton golden reference. `scripts/` exports weights, fixtures, and recurrent state and performs parity checks. `pretrain.py` is optional training infrastructure. Committed fixtures and expected JSON results live under `c_impl/fixtures_*` and `c_impl/results_*`. For decode work, prefer `c_impl/doc/decode_disaggregated_gemv.md`; parts of the READMEs describe the retired prefill flow.

## Build, Test, and Development Commands

- `make -C c_impl help` lists targets and configurable platform, frequency, and device knobs.
- `make -C c_impl` builds the native `gdn_eval` testbench with C++14. It requires Vitis HLS 2022.1 headers (`XILINX_HLS_INC`).
- `bash scripts/decode_correctness_check.sh --fast` runs the short exact-match decode gate; omit `--fast` for the full 32-step check.
- `cd c_impl && vitis_hls -f test.tcl` runs decode-kernel csim/csynth/cosim.
- `make -C c_impl run_hw` builds and runs the U55C hardware flow. Bitstream linking can take several hours and requires Vitis/XRT plus a board.
- For long-running builds, detach the build with a persistent PID, log, exit-code marker, and artifact paths so it survives chat interruption. Immediately attach a foreground watcher such as `tail --pid=<build-pid> -n 20 -F <active-run-log>` and leave that tool call pending and silent so Codex shows the running command, elapsed time, and live output. Follow the detailed Vivado `impl_1/runme.log` rather than a quiet milestone-only wrapper log. If the chat stops, only the watcher may end; reattach it later while the detached build continues.

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

# SLR2-region RTL cosim (one QKVG call)

Iter84 (2026-10-10). `make_cosim.sh <work dir>` copies `gdn_model.cpp`, a `gdn_model.h` whose
`GDN_COMPILED_WEIGHT_SHARD_BEATS` is scaled to 16,384 (one QKVG call's weights, the layer-0
state stripe and the probe tail fit a 20,992-beat port; the copy differs from the tree by
exactly that constant), `hls_gdn_forward.tcl`, the testbench and `cosim.tcl` into the work
dir; then `vitis_hls -f cosim.tcl` there (Vitis 2024.2, ~1 h 15 m on a build node: csynth of
the region + xsim). The top `gdn_slr2_region_cosim_top` (`-DGDN_SLR2_REGION_COSIM`, never part
of a build) calls `gdn_gemv_part_slr2` once for layer 0's QKVG call. The scalars make
beta = sigmoid(0) = 0.5 and decay = exp(-0) = 1 exactly, so the islands' arithmetic is plain
IEEE ops and cosim's C/RTL comparison of the ys words, attention beats and written state is
bit-exact; the AESL deadlock detector covers every process dependency of the region (invariant
24: csim cannot see a dataflow deadlock). Add `-DGDN_CALL_PROBE=1` to both cflags in
`cosim.tcl` to run the probed region and print the RTL counter's records (`PROBE_REC` lines).
Verdict: `C/RTL co-simulation finished: PASS`, and the C and RTL `SLR2_REGION_COSIM` digests
must match.

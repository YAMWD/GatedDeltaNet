# 32-Port GEMV Microbenchmark

This directory contains the routed U55C `gemv_full` milestone. It is a single
kernel with 32 independent 512-bit HBM weight ports, eight four-channel compute
clusters, SLR-local result collectors, and one S2MM output writer. Every compute
cluster consumes eight packs per port every eight cycles, sustaining an average
of one weight pack per port per cycle.

The floorplan places compute clusters `2/3/3` across SLR0/1/2. AXI adapters are
left at their natural HMSS placement; registered MM2S streams form the boundary
to remote compute. Activation and result streams use depth-64 BRAM FIFOs.

## Build and Test

Vitis, Vivado, and XRT 2022.1 are required. Run from this directory:

```sh
make csim_full
make host_full
make xo_full JOBS=8 FREQ=150 READ_OUTSTANDING=16
make xclbin_full JOBS=8 FREQ=150 READ_OUTSTANDING=16
```

Run the saturation benchmark and the real layer-0 `q_proj` check:

```sh
source /opt/xilinx/xrt/setup.sh
make run_full RUN_FREQ=130.6 TIMED_REPS=9
make run_layer0 RUN_FREQ=130.6 \
  WEIGHTS=../../artifacts/gdn-1.3b-f32.gdnw
```

`run_layer0` reads the `.gdnw` header, seeks directly to layer 0 `q_proj`, and
shards its 2,048 output rows across all 32 ports. Both runs compare hardware
output with a scalar GEMV using the same floating-point reduction order.

For long hardware links, launch `make xclbin_full` through
`run_build_with_marker.sh`, retain the PID, and watch Vivado's
`impl_1/runme.log` with `tail --pid=<pid> -F`.

## Routed Milestone

Measured on U55C with XRT 2022.1 on 2026-07-13:

| Check | Result |
|---|---:|
| Route errors / unrouted nets | 0 / 0 |
| Requested / achieved data clock | 150 / 130.6 MHz |
| Setup WNS / hold WHS | -0.985 / 0.002 ns |
| Saturation shape | 32 x 2,048 x 5,632 FP32 |
| Saturation bandwidth | **263.063 GB/s** |
| Saturation compute rate | **131.531 GFLOP/s** |
| Kernel-ceiling efficiency | **98.353%** |
| Synthetic parity | PASS, max abs 0 |
| Real layer-0 `q_proj` parity | PASS, max abs 0 |
| Real `q_proj` host-visible rate | 139.964 GB/s, 69.982 GFLOP/s |

The 16.8 MB real projection is short enough that XRT launch/completion overhead
reduces its host-visible rate. The saturation result represents the sustained
datapath rate.

Post-route analysis found SLR1 at 90.47% CLB and 96.88% BRAM utilization, with
95.03% of SLR0-SLR1 SLLs used. The worst setup path is 97.4% routing delay.
Consequently this design is the known-good 130.6 MHz baseline; the next timing
iteration should split the four-port clusters rather than replicate high-fanout
drivers in the congested implementation.

Generated `.xo`, `.xclbin`, DCP, report, executable, and log files are not
committed. Use `make distclean` to remove them.

# hbm_rd — the HBM pseudo-channel read ceiling (Iter88 diagnosis, 2026-10-14)

A read-only streaming kernel with one 512-bit `m_axi` port (4 KB bursts, 32 outstanding),
four compute units on SLR0 on HBM[0], [12], [20] and [31], linked at **300 MHz** so each port
asks for 19.2 GB/s, above the channel's 14.4 GB/s pin-rate peak. The host (XRT native, card by
PCIe BDF) streams 256 MiB × 16 passes per run, times each unit alone and all four together, and
checks a 64-bit two-lane sum. Not part of the production flow.

```bash
sbatch build.slurm                 # v++ at 300 MHz on the build partition (1.5 h); clocks forced and reported
sbatch card.slurm                  # harrier's U55C, builds the host there, runs single + concurrent
./host_hbm_rd.exe <xclbin> <bdf> [MiB=256] [reps=16] [rounds=3] [ncu=4]
```

Measured (build 10430, card job 10435, DATA_CLK 300 MHz, hbm_aclk 450 MHz read back):
**12.87 GB/s per channel**, alone or four at once (spread 0.02 %) = **89.4 % of 14.4 GB/s**;
32 channels sustain 411.8 GB/s. A 512-bit port reaches this ceiling at 201 MHz; the production
weight streams at 200 MHz (0.985 beats/cycle, 12.61 GB/s) run at 98 % of it. Details in
`doc/optimization_log.md`, Iter88.

#!/usr/bin/env python3
"""Iter81 pre-link gate, fail-closed, run on the build directory after the XO gate.

1. Island cell names. The link floorplan (apply_p_islands_body.tcl) puts each
   recurrent island into its quarter-SLR pblock by name and errors out about 50
   minutes into a 6-hour link if a name does not match exactly one cell. Check
   the names in the gdn_k_slr2 XO's RTL now: each must appear exactly once.
2. HLS 200-656 (an auto-rewind pipeline in a region without start
   propagation; invariant 24). Simulation cannot see the resulting hang. The
   production image (Iter79 P6, XO of 2026-10-03) carries exactly two such
   warnings, both on readers that write only region-internal streams:
   gdn_recurrent_duplicate_qkv (SLR2 islands region) and gdn_gemv_tiny_mm2s
   (control kernel's tiny a/b GEMV, `disable_start_propagation`). That image
   ran the 512-token drift and the 314,843-token WikiText runs without a hang,
   so these two are allowed; any other process named by 200-656 fails.
"""
import re
import sys
import zipfile
from pathlib import Path

EXPECTED_ISLANDS = ("gdn_recurrent_attention_island_p_0_U0",
                    "gdn_recurrent_attention_island_p_1_U0")
ALLOWED_656 = {"gdn_recurrent_duplicate_qkv", "gdn_gemv_tiny_mm2s"}
KERNELS = ("gdn_forward_p", "gdn_k_slr0", "gdn_k_slr2")


def main(build: Path) -> int:
    failures = []
    xo = build / "gdn_k_slr2.xo"
    counts = {name: 0 for name in EXPECTED_ISLANDS}
    if not xo.is_file():
        failures.append(f"missing {xo}")
    else:
        with zipfile.ZipFile(xo) as archive:
            rtl = [n for n in archive.namelist() if re.search(r"/hdl/verilog/[^/]+\.v$", n)]
            if not rtl:
                failures.append("no Verilog RTL inside gdn_k_slr2.xo")
            for member in rtl:
                text = archive.read(member).decode(errors="replace")
                for name in EXPECTED_ISLANDS:
                    counts[name] += len(re.findall(r"\b" + name + r"\b", text))
    for name, count in counts.items():
        print(f"ISLAND_NAME {name} occurrences={count}")
        if count != 1:
            failures.append(f"{name}: {count} occurrences in the RTL, expected exactly 1")
    for kernel in KERNELS:
        log = build / f"_x_compile_{kernel}" / kernel / kernel / "runme.log"
        if not log.is_file():
            failures.append(f"missing HLS log {log}")
            continue
        flagged = re.findall(r"HLS 200-656\][^\n]*?process (\S+) is instantiated",
                             log.read_text(errors="replace"))
        print(f"HLS_200_656 {kernel} processes={sorted(set(flagged))}")
        for process in flagged:
            if process not in ALLOWED_656:
                failures.append(f"{kernel}: HLS 200-656 names {process}")
    if failures:
        for failure in failures:
            print(f"ISLAND_NAME_GATE_FAIL {failure}")
        return 1
    print("ISLAND_NAME_GATE_PASS")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: check_island_names.py <build_dir>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))

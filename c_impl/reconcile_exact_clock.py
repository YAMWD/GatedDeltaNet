#!/usr/bin/env python3
"""Reconcile an XCLBIN's DATA_CLK metadata with the finishing hook's proven closure.

vpl decides AUTO-FREQ-SCALING from the timing report it writes *before* the
POST_ROUTE_PHYS_OPT_DESIGN.TCL.POST hook runs. finish_f150_timing.tcl closes the
design after that report (and aborts the link if it cannot), so the bitstream is
closed while the metadata still carries the scaled frequency -- Iter76 draw 4
(build 3987): design closed at 6.667 ns, xclbin said DATA_CLK = 149.

This tool patches DATA_CLK to the requested frequency ONLY when the hook's
exact_clock_gate.tsv proves every clock non-negative on setup and hold at the
requested period. Otherwise it changes nothing and exits non-zero. It never
touches the bitstream section.

usage: reconcile_exact_clock.py <xclbin> <exact_clock_gate.tsv> <link_freq_mhz> <workdir>
exit 0  EXACT_CLOCK_OK (already correct) or EXACT_CLOCK_RECONCILED (patched, re-read)
exit 1  EXACT_CLOCK_FAIL (no/negative gate evidence, or metadata still wrong)
"""
import json, math, os, shutil, subprocess, sys

XCLBINUTIL = os.environ.get("XCLBINUTIL", "/opt/xilinx/xrt/bin/xclbinutil")
KERNEL_CLOCK = "clk_kernel_00_unbuffered_net"
# Every clock the finishing hook gates, with its fixed period where the shell
# owns it. The kernel period comes from the requested frequency.
FIXED_CLOCKS = {"dma_ip_axi_aclk_1": 4.000, "hbm_aclk": 2.222}


def fail(msg):
    print(f"EXACT_CLOCK_FAIL: {msg}")
    sys.exit(1)


def dump_topology(xclbin, out_json):
    subprocess.run([XCLBINUTIL, "--input", xclbin, "--dump-section",
                    f"CLOCK_FREQ_TOPOLOGY:JSON:{out_json}", "--force", "--quiet"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return json.load(open(out_json))


def data_clocks(topology):
    return [float(c["m_freq_Mhz"]) for c in topology["clock_freq_topology"]["m_clock_freq"]
            if c["m_type"] == "DATA"]


def gate_proves_closure(tsv_path, link_freq):
    """True only if the hook's TSV exists, contains every required clock exactly
    once at its expected period, and every field is finite with setup >= 0 and
    hold >= 0. Anything malformed, missing or non-finite is a failure."""
    if not os.path.isfile(tsv_path):
        return False, f"gate evidence missing: {tsv_path}"
    rows = [l.rstrip("\n").split("\t") for l in open(tsv_path) if l.strip()]
    if not rows or rows[0][:4] != ["clock", "period_ns", "setup_wns_ns", "hold_whs_ns"]:
        return False, f"unexpected TSV header in {tsv_path}"
    expected = dict(FIXED_CLOCKS, **{KERNEL_CLOCK: 1000.0 / link_freq})
    seen = {}
    for r in rows[1:]:
        if len(r) < 4:
            return False, f"malformed row {r!r} in {tsv_path}"
        name = r[0]
        try:
            period, setup, hold = (float(x) for x in r[1:4])
        except ValueError:
            return False, f"non-numeric field in row {r!r}"
        if not all(math.isfinite(v) for v in (period, setup, hold)):
            return False, f"non-finite field in row {r!r}"
        if name not in expected:
            return False, f"unexpected clock {name} in {tsv_path}"
        if name in seen:
            return False, f"duplicate row for {name}"
        if abs(period - expected[name]) > 0.002:
            return False, f"{name} period {period} ns is not {expected[name]:.3f} ns"
        if setup < 0 or hold < 0:
            return False, f"{name} setup={setup} hold={hold} at {period} ns is negative"
        seen[name] = (setup, hold)
    missing = sorted(set(expected) - set(seen))
    if missing:
        return False, f"required clock(s) missing from {tsv_path}: {missing}"
    return True, (f"gate: {len(seen)} clocks non-negative -- "
                  + ", ".join(f"{n} {s:+.3f}/{h:+.3f}" for n, (s, h) in sorted(seen.items()))
                  + f"; kernel at {expected[KERNEL_CLOCK]:.3f} ns")


def main():
    if len(sys.argv) != 5:
        fail(__doc__)
    xclbin, tsv, link_freq, work = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    os.makedirs(work, exist_ok=True)
    before = dump_topology(xclbin, os.path.join(work, "clock_freq_topology.json"))
    clocks = data_clocks(before)
    print(f"image DATA clocks: {clocks}  requested: {link_freq}")
    # Closure evidence is required in every case: an image whose metadata already
    # says the requested frequency is only acceptable if the hook proved timing.
    ok, why = gate_proves_closure(tsv, link_freq)
    print(why)
    if not ok:
        fail(f"DATA={clocks}, requested={link_freq}, and closure is not proven -- not accepting")
    if clocks == [link_freq]:
        print(f"EXACT_CLOCK_OK {clocks} (closure proven by the hook gate)")
        return
    entries = [c for c in before["clock_freq_topology"]["m_clock_freq"] if c["m_type"] == "DATA"]
    if len(entries) != 1:
        fail(f"expected one DATA clock entry, found {len(entries)}")
    old = entries[0]["m_freq_Mhz"]
    entries[0]["m_freq_Mhz"] = str(int(link_freq)) if link_freq.is_integer() else str(link_freq)
    patched_json = os.path.join(work, "clock_freq_topology_reconciled.json")
    json.dump(before, open(patched_json, "w"), indent=4)
    tmp = xclbin + ".reconciled.tmp"
    subprocess.run([XCLBINUTIL, "--input", xclbin, "--replace-section",
                    f"CLOCK_FREQ_TOPOLOGY:JSON:{patched_json}", "--output", tmp, "--force", "--quiet"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    after = data_clocks(dump_topology(tmp, os.path.join(work, "clock_freq_topology_after.json")))
    if after != [link_freq]:
        os.remove(tmp)
        fail(f"patch did not take: DATA={after}")
    shutil.copyfile(xclbin, xclbin + ".vpl_metadata.bak")
    os.replace(tmp, xclbin)
    print(f"EXACT_CLOCK_RECONCILED: DATA_CLK {old} -> {entries[0]['m_freq_Mhz']} "
          f"(vpl scaled from a pre-hook timing report; the hook's gate proves closure). "
          f"Original kept as {os.path.basename(xclbin)}.vpl_metadata.bak")


if __name__ == "__main__":
    main()

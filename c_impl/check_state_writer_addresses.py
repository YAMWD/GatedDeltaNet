#!/usr/bin/env python3
"""Test actual XO writers before/after Vivado OOC synthesis, before full link.

Exercises all 24 production layer offsets and two physical base addresses per
port. This tests the internal HLS word-address request channel, not a complete
AXI adapter or post-placement design; on-card state/logit checks remain required.
"""
import argparse
import json
import os
import re
import subprocess
import zipfile
from pathlib import Path


def command(args, cwd):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def testbench(top, ports, port, allow_direct=False):
    names = {name for direction, width, name in ports}
    def one(pattern):
        found = [n for n in names if re.fullmatch(pattern, n)]
        if len(found) != 1:
            raise ValueError(f"Expected one {pattern}, got {found}")
        return found[0]
    suffix = r"(?:_dout)?" if allow_direct else r"_dout"
    pointer = one(r"(?:w" + str(port) + r"|weights)" + suffix)
    layer = one(r"layer_index" + suffix)
    mode = one(r"qkvg_recurrent_mode" + suffix)
    addr = one(r"m_axi_.*_AWADDR")
    valid = one(r"m_axi_.*_AWVALID")
    declarations = [f"{'reg' if d == 'input' else 'wire'} {w + ' ' if w else ''}{n};" for d, w, n in ports]
    initialize = []
    for direction, _, name in ports:
        if direction != "input" or name == "ap_clk":
            continue
        value = "1" if name.endswith(("_empty_n", "_TVALID", "_AWREADY", "_WREADY", "_BVALID")) or name == "ap_continue" else "0"
        if name.endswith(("_num_data_valid", "_fifo_cap")):
            value = "4096" if name.startswith("state_wr") else "1"
        initialize.append(f"{name} = {value};")
    return "\n".join([
        "`timescale 1ns/1ps", "module address_tb;", *declarations,
        "integer li, relocation, cycles, checks, failures; reg seen; reg [63:0] expected;",
        "initial ap_clk=0; always #3.333 ap_clk=~ap_clk;",
        top + " dut(" + ",".join(f".{n}({n})" for _, _, n in ports) + ");",
        "initial begin", *initialize, "checks=0; failures=0;",
        # Allow the post-synthesis global reset (glbl) to deassert.
        "#200;",
        "for(relocation=0; relocation<2; relocation=relocation+1) begin",
        "for(li=0; li<24; li=li+1) begin",
        "@(negedge ap_clk); ap_start=0; ap_rst=1;",
        "repeat(8) @(negedge ap_clk);",
        f"{pointer}=64'h{port * 0x20000000:x} + (relocation ? 64'h8000000 : 0);",
        f"{layer}=li; {mode}=1;",
        f"expected=({pointer}+64'd87457792+64'd262144*li)>>6;",
        "ap_rst=0; ap_start=1; seen=0;",
        "begin : await_address",
        "for(cycles=0; cycles<512; cycles=cycles+1) begin",
        "@(posedge ap_clk);",
        f"if({valid}) begin",
        f'if({addr} !== expected) begin $display("ADDRESS_FAIL port={port} layer=%0d pointer=%h expected_word=%h actual_word=%h",li,{pointer},expected,{addr}); failures=failures+1; end',
        "seen=1; checks=checks+1; disable await_address; end",
        "end end",
        'if(!seen) begin $display("ADDRESS_TIMEOUT"); failures=failures+1; end',
        "end end",
        f'if(failures==0 && checks==48) $display("ADDRESS_PASS port={port} cases=%0d",checks);',
        f'else $display("ADDRESS_FAILED_SUMMARY port={port} cases=%0d failures=%0d",checks,failures);',
        "$finish;",
        "end", "endmodule", "",
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xo", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--ports", type=int, nargs="+", default=[28, 29, 30, 31], choices=[28, 29, 30, 31])
    parser.add_argument("--diagnostic-continue", action="store_true",
                        help="Collect all phase verdicts even after an address mismatch; still exit nonzero")
    parser.add_argument("--diagnostic-direct-inputs", action="store_true",
                        help="Accept scalar rather than FIFO control inputs in an isolated HLS wrapper")
    parser.add_argument("--diagnostic-keep-offset", action="store_true",
                        help="Test KEEP on the narrow offset net; never modifies the input XO")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.xo.is_dir():
        sources = {p.stem: p.read_text() for p in args.xo.glob("gdn_forward*.v")}
    else:
        with zipfile.ZipFile(args.xo) as archive:
            sources = {Path(n).stem: archive.read(n).decode() for n in archive.namelist()
                       if "/hdl/verilog/" in n and n.endswith(".v")}
    results = []
    for port in args.ports:
        tops = [n for n in sources if re.fullmatch(r"gdn_forward_gemv32_state_writer_" + str(port) + r"_s", n)]
        if len(tops) != 1:
            raise ValueError(f"Expected one writer{port}, got {tops}")
        top = tops[0]
        if args.diagnostic_keep_offset:
            sources[top], count = re.subn(
                r"(?m)^(wire\s+\[26:0\]\s+add_ln\w+;)$",
                r'(* keep = "true" *) \1', sources[top])
            if count != 1:
                raise ValueError(f"Expected one 27-bit offset net, got {count}")
        work = out / f"port{port}"
        work.mkdir(exist_ok=True)
        selected, todo = {}, [top]
        while todo:
            name = todo.pop()
            if name in selected:
                continue
            selected[name] = sources[name]
            words = set(re.findall(r"\b[a-zA-Z_]\w*\b", sources[name]))
            todo.extend(n for n in words.intersection(sources) if n not in selected)
        for name, text in selected.items():
            (work / (name + ".v")).write_text(text)
        ports = re.findall(r"^(input|output)\s+(\[\d+:\d+\])?\s*(\w+)\s*;", sources[top], re.M)
        if not ports:
            raise ValueError("Cannot parse writer ports")
        phases = (("rtl", top), ("synth", "writer_synth"), ("opt", "writer_post"))
        for phase, dut in phases:
            (work / f"tb_{phase}.sv").write_text(testbench(dut, ports, port, args.diagnostic_direct_inputs))
        files = [work / (n + ".v") for n in selected]
        tcl = "\n".join([
            "read_verilog [list " + " ".join("{" + str(f) + "}" for f in files) + "]",
            f"synth_design -top {top} -part xcu55c-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy rebuilt",
            "create_clock -period 6.667 [get_ports ap_clk]",
            "write_verilog -force -mode funcsim -rename_top writer_synth writer_synth.v",
            "write_checkpoint -force writer_synth.dcp",
            "opt_design",
            "write_verilog -force -mode funcsim -rename_top writer_post writer_post.v",
            "write_checkpoint -force writer_post.dcp", "exit 0", "",
        ])
        (work / "synth.tcl").write_text(tcl)
        command(["vivado", "-mode", "batch", "-nojournal", "-log", "synth.log", "-source", "synth.tcl"], work)
        for phase, dut in phases:
            rtl = files if phase == "rtl" else [work / (dut + ".v")]
            if phase == "rtl":
                glbl = Path(os.environ.get("XILINX_VIVADO", "/tools/Xilinx/Vivado/2024.2")) / "data/verilog/src/glbl.v"
                command(["xvlog", glbl], work)
            command(["xvlog", "--sv", *rtl, work / f"tb_{phase}.sv"], work)
            command(["xelab", "address_tb", "glbl", "-L", "unisims_ver", "-s", f"sim_{phase}"], work)
            command(["xsim", f"sim_{phase}", "-runall", "-log", f"sim_{phase}.log"], work)
            log = (work / f"sim_{phase}.log").read_text()
            passed = (f"ADDRESS_PASS port={port} cases=48" in log and
                      "ADDRESS_FAIL" not in log and "ADDRESS_TIMEOUT" not in log)
            results.append(dict(port=port, phase=phase, cases=48, passed=passed))
            (out / "summary.json").write_text(json.dumps(results, indent=2))
            if not passed and not args.diagnostic_continue:
                raise ValueError(f"{port} {phase}: address test did not pass")
    (out / "summary.json").write_text(json.dumps(results, indent=2))
    if not all(r["passed"] for r in results):
        raise ValueError("State address gate failed; see summary.json")
    count = 48 * len(args.ports)
    print(f"STATE_ADDRESS_GATE_PASS rtl_cases={count} synthesized_cases={count} optimized_cases={count}", flush=True)


if __name__ == "__main__":
    main()

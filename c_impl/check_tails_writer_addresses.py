#!/usr/bin/env python3
"""Iter86: synthesized-address gate for the control kernel's conv-tail writer (gdn_tails_writer, mm0).

The same method as check_state_writer_addresses.py, for the writer the Iter86 control region
puts on the workspace bundle: the HLS RTL of the writer module (from a csynth verilog directory
or the XO) is simulated with xsim at three stages -- HLS RTL, after Vivado out-of-context
synth_design, after opt_design -- over all 24 layer offsets and three base addresses, and every
AXI write request at the module boundary must land at the expected 64-byte word address of the
next tail beat in stream order (head-major, then kind, row, beat: base + head_buf +
(layer*3 + kind)*384 + row*64 + head*8 + p), with 576 beats requested and handed over per call,
followed (probe builds) by the four counter record beats at probe_off + call*4. Invariant 26:
Vivado synthesis folded a narrower signed offset 128 MiB low on the state writers (Iter75) and
readers (Iter81) with correct HLS RTL; this writer's RTL carries the same 26-bit signed offset
pattern, so the three stages are compared, not assumed.
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


def layout_constants(c_impl):
    """GDN_WS_OFF_HEAD_BUF / 16 and GDN_WS_PROBE_BEAT_OFF, compiled from the header so the
    gate can never drift from the kernel's workspace layout."""
    import tempfile
    include = os.environ.get("XILINX_HLS_INC", "/tools/Xilinx/Vitis_HLS/2022.1/include")
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "ws_consts.cpp"
        src.write_text('#include <cstdio>\n#include <cstdint>\n#include "gdn_model.h"\n'
                       'int main() { printf("%llu %llu\\n", (unsigned long long)(GDN_WS_OFF_HEAD_BUF / 16), '
                       '(unsigned long long)GDN_WS_PROBE_BEAT_OFF); return 0; }\n')
        exe = Path(tmp) / "ws_consts"
        subprocess.run(["c++", "-std=c++14", f"-I{c_impl}", f"-I{include}", "-o", str(exe), str(src)], check=True)
        head_buf, probe_off = subprocess.run([str(exe)], check=True, capture_output=True, text=True).stdout.split()
    return int(head_buf), int(probe_off)


def testbench(top, ports, head_buf, probe_off, relocations):
    names = {name for direction, width, name in ports}

    def one(pattern):
        found = [n for n in names if re.fullmatch(pattern, n)]
        if len(found) != 1:
            raise ValueError(f"Expected one {pattern}, got {found}")
        return found[0]

    def first_of(*patterns):
        for pattern in patterns:
            found = [n for n in names if re.fullmatch(pattern, n)]
            if len(found) == 1:
                return found[0]
        raise ValueError(f"Expected one of {patterns}, got none (or several)")

    pointer = first_of(r"workspace(?:_c)?_dout", r"workspace")
    layer = first_of(r"layer_index_dout", r"layer_index")
    mode = first_of(r"qkvg_recurrent_mode_dout", r"qkvg_recurrent_mode", r"enabled_dout", r"enabled")
    probe = any(re.fullmatch(r"probe_call(?:_dout)?", n) for n in names)
    call = first_of(r"probe_call_dout", r"probe_call") if probe else None
    addr = one(r"m_axi_.*_AWADDR")
    valid = one(r"m_axi_.*_AWVALID")
    awready = one(r"m_axi_.*_AWREADY")
    awlen = one(r"m_axi_.*_AWLEN")
    wvalid = one(r"m_axi_.*_WVALID")
    wready = one(r"m_axi_.*_WREADY")
    tail_beats, probe_beats = 576, (4 if probe else 0)
    declarations = [f"{'reg' if d == 'input' else 'wire'} {w + ' ' if w else ''}{n};" for d, w, n in ports]
    initialize = []
    for direction, _, name in ports:
        if direction != "input" or name == "ap_clk":
            continue
        value = "1" if name.endswith(("_empty_n", "_full_n", "_TVALID", "_AWREADY", "_WREADY", "_BVALID")) or name == "ap_continue" else "0"
        if name.endswith(("_num_data_valid", "_fifo_cap")):
            value = "1"
        if name.endswith("tails_stream_dout"):
            value = "512'h5a5a"
        initialize.append(f"{name} = {value};")
    cases = []  # (relocation, layer, mode, call, total beats, beat offset into the expected sequence)
    for relocation in relocations:
        for li in range(24):
            cases.append((relocation, li, 1, li % 97, tail_beats + probe_beats, 0))
        if probe:
            cases.append((relocation, 0, 0, 96, probe_beats, tail_beats))
    case_rows = [f"begin reloc=64'h{r:x}; li={li}; md={md}; cl={cl}; total={tot}; off={off}; end" for r, li, md, cl, tot, off in cases]
    return "\n".join([
        "`timescale 1ns/1ps", "module address_tb;", *declarations,
        "integer ci, li, md, cl, cycles, checks, failures, requests, total, off, extra; reg seen;",
        "reg [63:0] reloc, expected, req_beats, data_beats, base_word;",
        "function automatic [63:0] exp_word(input [63:0] bw, input integer layer, input integer pc, input integer n);",
        "  integer h, k, r, p;",
        "  begin",
        f"    if (n < {tail_beats}) begin",
        "      h = n / 72; k = (n % 72) / 24; r = (n % 24) / 8; p = n % 8;",
        f"      exp_word = bw + 64'd{head_buf} + (layer * 3 + k) * 384 + r * 64 + h * 8 + p;",
        "    end else begin",
        f"      exp_word = bw + 64'd{probe_off} + pc * 4 + (n - {tail_beats});",
        "    end",
        "  end",
        "endfunction",
        "initial ap_clk=0; always #2.5 ap_clk=~ap_clk;",
        top + " dut(" + ",".join(f".{n}({n})" for _, _, n in ports) + ");",
        # ap_ctrl handshake as the dataflow region drives it: start is held only until the
        # module's ap_ready, then dropped (the region's ap_sync logic does the same). Holding
        # it would restart the module on its idle state with the same inputs -- a second run
        # the reset then aborts with stale datapath state (config_rtl -reset control), which
        # corrupted every case after a clean one in the first probe-free run (job 8228).
        "always @(negedge ap_clk) if (ap_ready === 1'b1) ap_start = 0;",
        "initial begin", *initialize, "checks=0; failures=0;",
        "#200;",
        f"for(ci=0; ci<{len(cases)}; ci=ci+1) begin",
        "case(ci)",
        *[f"{i}: {row}" for i, row in enumerate(case_rows)],
        "default: begin reloc=0; li=0; md=0; cl=0; total=0; off=0; end",
        "endcase",
        "@(negedge ap_clk); ap_start=0; ap_rst=1;",
        "repeat(8) @(negedge ap_clk);",
        f"{pointer}=reloc; base_word=reloc>>6;",
        f"{layer}=li; {mode}=md;" + (f" {call}=cl;" if call else ""),
        "ap_rst=0; ap_start=1; seen=0; requests=0; req_beats=0; data_beats=0;",
        "begin : await_address",
        "for(cycles=0; cycles<32768; cycles=cycles+1) begin",
        "@(posedge ap_clk);",
        f"if({valid} && {awready}) begin",
        f"expected = exp_word(base_word, li, cl, off + req_beats);",
        f'if({addr} !== expected) begin $display("ADDRESS_FAIL case=%0d layer=%0d mode=%0d pointer=%h request=%0d beat=%0d expected_word=%h actual_word=%h",ci,li,md,{pointer},requests,off+req_beats,expected,{addr}); failures=failures+1; end',
        f"req_beats = req_beats + {awlen}; requests = requests + 1; seen=1; end",
        f"if({wvalid} && {wready}) data_beats = data_beats + 1;",
        "if(seen && req_beats >= total && data_beats >= total) disable await_address;",
        "end end",
        # no write request may follow a completed case (a restart or a stray burst)
        "extra = 0; repeat(256) begin @(posedge ap_clk); "
        f"if({valid} && {awready}) extra = extra + 1; end",
        'if(extra != 0) begin $display("EXTRA_REQUEST_FAIL case=%0d layer=%0d mode=%0d extra_requests=%0d",ci,li,md,extra); failures=failures+1; end',
        'if(ap_start !== 1\'b0) begin $display("START_NOT_ACKED case=%0d layer=%0d",ci,li); failures=failures+1; end',
        f'if(!seen) begin $display("ADDRESS_TIMEOUT case=%0d layer=%0d mode=%0d",ci,li,md); failures=failures+1; end',
        "else begin",
        f'if(req_beats !== total || data_beats !== total) begin $display("LENGTH_FAIL case=%0d layer=%0d mode=%0d requests=%0d requested_beats=%0d data_beats=%0d total=%0d",ci,li,md,requests,req_beats,data_beats,total); failures=failures+1; end',
        "checks=checks+1; end",
        "end",
        f'if(failures==0 && checks=={len(cases)}) $display("TAILS_ADDRESS_PASS cases=%0d tail_beats_per_call={tail_beats} probe_beats_per_call={probe_beats}",checks);',
        f'else $display("TAILS_ADDRESS_FAILED_SUMMARY cases=%0d failures=%0d",checks,failures);',
        "$finish;",
        "end", "endmodule", "",
    ]), len(cases)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("xo", type=Path, help="gdn_forward_p.xo, or a directory of its HLS verilog")
    parser.add_argument("out", type=Path)
    parser.add_argument("--head-buf-beats", type=int, help="GDN_WS_OFF_HEAD_BUF / 16 (default: compiled from gdn_model.h)")
    parser.add_argument("--probe-off-beats", type=int, help="GDN_WS_PROBE_BEAT_OFF (default: compiled from gdn_model.h)")
    parser.add_argument("--c-impl", type=Path, default=Path(__file__).resolve().parent,
                        help="directory holding gdn_model.h, for the two layout constants")
    parser.add_argument("--print-constants", action="store_true", help="derive the constants and exit")
    parser.add_argument("--relocations", type=lambda s: int(s, 0), nargs="+", default=[0, 0x8000000, 0x100000000])
    parser.add_argument("--diagnostic-continue", action="store_true")
    parser.add_argument("--clock-ns", type=float, default=5.0)
    args = parser.parse_args()
    if args.head_buf_beats is None or args.probe_off_beats is None:
        head_buf, probe_off = layout_constants(args.c_impl)
        args.head_buf_beats = head_buf if args.head_buf_beats is None else args.head_buf_beats
        args.probe_off_beats = probe_off if args.probe_off_beats is None else args.probe_off_beats
    print(f"layout: head_buf_beats={args.head_buf_beats} probe_off_beats={args.probe_off_beats}", flush=True)
    if args.print_constants:
        return
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.xo.is_dir():
        sources = {p.stem: p.read_text() for p in args.xo.glob("*.v")}
    else:
        with zipfile.ZipFile(args.xo) as archive:
            sources = {Path(n).stem: archive.read(n).decode() for n in archive.namelist()
                       if "/hdl/verilog/" in n and n.endswith(".v")}
    tops = [n for n in sources if re.fullmatch(r"gdn_forward_p_gdn_tails_writer", n)]
    if len(tops) != 1:
        raise ValueError(f"Expected one tails writer, got {tops}")
    top = tops[0]
    work = out / "tails_writer"
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
    cases = None
    for phase, dut in phases:
        text, cases = testbench(dut, ports, args.head_buf_beats, args.probe_off_beats, args.relocations)
        (work / f"tb_{phase}.sv").write_text(text)
    files = [work / (n + ".v") for n in selected]
    tcl = "\n".join([
        "read_verilog [list " + " ".join("{" + str(f) + "}" for f in files) + "]",
        f"synth_design -top {top} -part xcu55c-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy rebuilt",
        f"create_clock -period {args.clock_ns:.3f} [get_ports ap_clk]",
        "write_verilog -force -mode funcsim -rename_top writer_synth writer_synth.v",
        "write_checkpoint -force writer_synth.dcp",
        "opt_design",
        "write_verilog -force -mode funcsim -rename_top writer_post writer_post.v",
        "write_checkpoint -force writer_post.dcp", "exit 0", "",
    ])
    (work / "synth.tcl").write_text(tcl)
    command(["vivado", "-mode", "batch", "-nojournal", "-log", "synth.log", "-source", "synth.tcl"], work)
    results = []
    for phase, dut in phases:
        rtl = files if phase == "rtl" else [work / (dut + ".v")]
        if phase == "rtl":
            glbl = Path(os.environ.get("XILINX_VIVADO", "/tools/Xilinx/Vivado/2024.2")) / "data/verilog/src/glbl.v"
            command(["xvlog", glbl], work)
        command(["xvlog", "--sv", *rtl, work / f"tb_{phase}.sv"], work)
        command(["xelab", "address_tb", "glbl", "-L", "unisims_ver", "-s", f"sim_{phase}"], work)
        command(["xsim", f"sim_{phase}", "-runall", "-log", f"sim_{phase}.log"], work)
        log = (work / f"sim_{phase}.log").read_text()
        passed = (f"TAILS_ADDRESS_PASS cases={cases}" in log and "ADDRESS_FAIL" not in log
                  and "ADDRESS_TIMEOUT" not in log and "LENGTH_FAIL" not in log
                  and "EXTRA_REQUEST_FAIL" not in log and "START_NOT_ACKED" not in log)
        results.append(dict(phase=phase, cases=cases, passed=passed))
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        if not passed and not args.diagnostic_continue:
            raise ValueError(f"{phase}: tails address test did not pass")
    if not all(r["passed"] for r in results):
        raise ValueError("Tails writer address gate failed; see summary.json")
    print(f"TAILS_ADDRESS_GATE_PASS rtl_cases={cases} synthesized_cases={cases} optimized_cases={cases}", flush=True)


if __name__ == "__main__":
    main()

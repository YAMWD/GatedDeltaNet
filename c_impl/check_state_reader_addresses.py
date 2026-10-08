#!/usr/bin/env python3
"""Iter81 pre-link gate (fail-closed): the state READER's synthesized address, the twin of
check_state_writer_addresses.py. Simulates gemv32_mm2s_with_state_p_<port>
alone at HLS RTL, after Vivado OOC synthesis and after opt_design, for all 24
layers and two base addresses. In QKVG mode the reader must first request the
layer's 4,096 state beats at w + 87,457,792 + 262,144*layer bytes and hand 4,096
beats to the state queue. The module's internal AR channel carries word
(64-byte) addresses, as the writer's AW channel does."""
import argparse, json, os, re, subprocess, zipfile
from pathlib import Path

def command(args, cwd):
    print("RUN", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)

def testbench(top, ports, port):
    names = {n for _, _, n in ports}
    def one(p):
        f = [n for n in names if re.fullmatch(p, n)]
        if len(f) != 1: raise ValueError(f"Expected one {p}, got {f}")
        return f[0]
    pointer, layer, mode = one(rf"w{port}(?:_dout)?"), one(r"layer_index(?:_dout)?"), one(r"qkvg_recurrent_mode(?:_dout)?")
    npacks = one(r"p_read(?:_dout)?"); off = one(r"w_pack_off(?:_dout)?")
    araddr, arvalid, arready, arlen = one(r"m_axi_.*_ARADDR"), one(r"m_axi_.*_ARVALID"), one(r"m_axi_.*_ARREADY"), one(r"m_axi_.*_ARLEN")
    swrite = one(r"state_stream\d_write")
    decl = [f"{'reg' if d == 'input' else 'wire'} {w + ' ' if w else ''}{n};" for d, w, n in ports]
    init = []
    for d, _, n in ports:
        if d != "input" or n == "ap_clk": continue
        v = "1" if n.endswith(("_full_n", "_ARREADY", "_RVALID", "_AWREADY", "_WREADY", "_BVALID")) or n == "ap_continue" else "0"
        if n.endswith("_RFIFONUM"): v = "9'd256"
        if n.endswith("_fifo_cap"): v = "4096" if n.startswith("state_stream") else "64"
        init.append(f"{n} = {v};")
    return "\n".join([
        "`timescale 1ns/1ps", "module address_tb;", *decl,
        "integer li, relocation, cycles, checks, failures, requests; reg seen;",
        "reg [63:0] expected, req_beats, data_beats;",
        "initial ap_clk=0; always #2.5 ap_clk=~ap_clk;",
        top + " dut(" + ",".join(f".{n}({n})" for _, _, n in ports) + ");",
        "initial begin", *init, "checks=0; failures=0;", "#200;",
        "for(relocation=0; relocation<2; relocation=relocation+1) begin",
        "for(li=0; li<24; li=li+1) begin",
        "@(negedge ap_clk); ap_start=0; ap_rst=1;", "repeat(8) @(negedge ap_clk);",
        f"{pointer}=64'h{port * 0x20000000:x} + (relocation ? 64'h8000000 : 0);",
        f"{layer}=li; {mode}=1; {npacks}=16384; {off}=0;",
        f"expected=({pointer}+64'd87457792+64'd262144*li)>>6;",
        "ap_rst=0; ap_start=1; seen=0; requests=0; req_beats=0; data_beats=0;",
        "begin : await_address",
        "for(cycles=0; cycles<40000; cycles=cycles+1) begin",
        "@(posedge ap_clk);",
        # A request inside the weight shard ([w, w + 1,366,528 beats)) is a weight
        # burst (the production reader interleaves them); any other is a state burst.
        f"if({arvalid} && {arready} && !({araddr} >= ({pointer}>>6) && {araddr} < ({pointer}>>6) + 64'd1366528)) begin",
        f'if({araddr} !== expected + req_beats) begin $display("ADDRESS_FAIL port={port} layer=%0d base=%h request=%0d expected_word=%h actual_word=%h len=%0d delta_bytes=%0d",li,{pointer},requests,expected+req_beats,{araddr},{arlen},($signed({araddr})-$signed(expected+req_beats))*64); failures=failures+1; end',
        f"req_beats = req_beats + {arlen}; requests = requests + 1; seen=1; end",
        f"if({swrite}) data_beats = data_beats + 1;",
        "if(seen && req_beats >= 4096 && data_beats >= 4096) disable await_address;",
        "end end",
        f'if(!seen) begin $display("ADDRESS_TIMEOUT port={port} layer=%0d",li); failures=failures+1; end',
        f'else if(req_beats !== 4096 || data_beats !== 4096) begin $display("LENGTH_FAIL port={port} layer=%0d requests=%0d requested_beats=%0d state_beats=%0d",li,requests,req_beats,data_beats); failures=failures+1; end',
        "checks=checks+1;", "end end",
        f'if(failures==0 && checks==48) $display("READER_ADDRESS_PASS port={port} cases=%0d",checks);',
        f'else $display("READER_ADDRESS_FAILED_SUMMARY port={port} cases=%0d failures=%0d",checks,failures);',
        "$finish;", "end", "endmodule", ""])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path); ap.add_argument("out", type=Path)
    ap.add_argument("--ports", type=int, nargs="+", default=[28, 29, 30, 31])
    ap.add_argument("--clock-ns", type=float, default=5.0)
    a = ap.parse_args(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    if a.src.is_dir():
        sources = {p.stem: p.read_text() for p in a.src.glob("*.v")}
    else:
        with zipfile.ZipFile(a.src) as z:
            sources = {Path(n).stem: z.read(n).decode() for n in z.namelist() if "/hdl/verilog/" in n and n.endswith(".v")}
    results = []
    for port in a.ports:
        tops = [n for n in sources if re.fullmatch(rf"gdn_k_slr2_gemv32_mm2s_with_state(?:_p)?_{port}_s", n)]
        if len(tops) != 1: raise ValueError(f"Expected one reader{port}, got {tops}")
        top = tops[0]; work = out / f"port{port}"; work.mkdir(exist_ok=True)
        sel, todo = {}, [top]
        while todo:
            n = todo.pop()
            if n in sel: continue
            sel[n] = sources[n]
            todo.extend(m for m in set(re.findall(r"\b[a-zA-Z_]\w*\b", sources[n])).intersection(sources) if m not in sel)
        for n, t in sel.items(): (work / (n + ".v")).write_text(t)
        ports = re.findall(r"^(input|output)\s+(\[\d+:\d+\])?\s*(\w+)\s*;", sources[top], re.M)
        phases = (("rtl", top), ("synth", "reader_synth"), ("opt", "reader_post"))
        for ph, dut in phases: (work / f"tb_{ph}.sv").write_text(testbench(dut, ports, port))
        files = [work / (n + ".v") for n in sel]
        (work / "synth.tcl").write_text("\n".join([
            "read_verilog [list " + " ".join("{" + str(f) + "}" for f in files) + "]",
            f"synth_design -top {top} -part xcu55c-fsvh2892-2L-e -mode out_of_context -flatten_hierarchy rebuilt",
            f"create_clock -period {a.clock_ns:.3f} [get_ports ap_clk]",
            "write_verilog -force -mode funcsim -rename_top reader_synth reader_synth.v",
            "opt_design",
            "write_verilog -force -mode funcsim -rename_top reader_post reader_post.v", "exit 0", ""]))
        command(["vivado", "-mode", "batch", "-nojournal", "-log", "synth.log", "-source", "synth.tcl"], work)
        glbl = Path(os.environ.get("XILINX_VIVADO", "/tools/Xilinx/Vivado/2024.2")) / "data/verilog/src/glbl.v"
        command(["xvlog", glbl], work)
        for ph, dut in phases:
            rtl = files if ph == "rtl" else [work / (dut + ".v")]
            command(["xvlog", "--sv", *rtl, work / f"tb_{ph}.sv"], work)
            command(["xelab", "address_tb", "glbl", "-L", "unisims_ver", "-s", f"sim_{ph}"], work)
            command(["xsim", f"sim_{ph}", "-runall", "-log", f"sim_{ph}.log"], work)
            log = (work / f"sim_{ph}.log").read_text()
            ok = f"READER_ADDRESS_PASS port={port} cases=48" in log
            fails = [l for l in log.splitlines() if re.match(r"(ADDRESS_FAIL|ADDRESS_TIMEOUT|LENGTH_FAIL)", l)]
            results.append(dict(port=port, phase=ph, passed=ok, first_failures=fails[:3], failure_lines=len(fails)))
            (out / "summary.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    passed = bool(results) and all(r["passed"] for r in results)
    print("READER_GATE_PASS" if passed else "READER_GATE_FAIL", flush=True)
    raise SystemExit(0 if passed else 1)

if __name__ == "__main__":
    main()

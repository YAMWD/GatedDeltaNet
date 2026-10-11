#!/usr/bin/env python3
"""Iter82 dual-kernel call probe (GDN_CALL_PROBE builds, host.exe --dump-probe): per-call timelines of the control kernel (1) and
the SLR2 kernel (2), their per-call alignment, and the token's decomposition.
Dump line: step kernel call record event port tag cycle (cycle from the call
region's start on that kernel).  Record slots are fixed (see gdn_model.cpp,
gdn_ctrl_probe_counter / gdn_call_probe_counter)."""
import json, sys, statistics as st
from collections import defaultdict

CTRL = {0: "x_bcast_done", 1: "w0_done", 2: "tx2_first", 3: "tx2_last", 4: "rx0_first", 5: "rx0_last",
        6: "rx2_first", 7: "rx2_last", 8: "col_first0", 9: "col_first1", 10: "col_first2", 11: "col_first3",
        12: "col_last0", 13: "col_last1", 14: "col_last2", 15: "col_last3",
        16: "col_sum0", 17: "col_sum1", 18: "col_sum2", 19: "col_sum3",
        20: "fill_done|ing_h0", 21: "out_done|ing_h7", 22: "con_h0", 23: "con_h7",
        24: "conv_h0k0", 25: "conv_h0k1", 26: "conv_h0k2", 27: "conv_h7k2", 28: "conv_tx_last",
        29: "attn_rx_first", 30: "attn_rx_last", 31: "end"}
SLR2 = {0: "xr_first", 1: "xr_last", 2: "ys_first", 3: "ys_last", 4: "attn_first", 5: "attn_last",
        6: "qkv_first", 7: "qkv_last", 8: "pf28", 9: "wt28", 10: "pf29", 11: "wt29", 12: "pf30", 13: "wt30",
        14: "pf31", 15: "wt31", 16: "wr28", 17: "wr29", 18: "wr30", 19: "wr31",
        20: "i0_prep_h0", 21: "i0_read_h0", 22: "i0_read_hl", 23: "i0_upd_hl", 24: "i0_upd_end",
        25: "i1_prep_h0", 26: "i1_read_h0", 27: "i1_read_hl", 28: "i1_upd_hl", 29: "i1_upd_end", 30: "go_wait", 31: "end"}
COL_WORDS = [8, 64, 24, 32]   # words per sub-loop per row group: c0 | slr0 | local b | slr2
def kind(call): return "lm_head" if call == 96 else ["QKVG", "o_proj", "GU", "down"][call % 4]
def row_groups(call): return {"QKVG": 32, "o_proj": 8, "GU": 44, "down": 8, "lm_head": 125}[kind(call)]
def med(xs): return st.median(xs) if xs else float("nan")

rows = [l.split() for l in open(sys.argv[1]) if not l.startswith("#")]
ex = json.load(open(sys.argv[2]))["examples"][0]
kernel_ms = [v for v in ex["kernel_ms"] if v > 0]
tok = med(kernel_ms) * 200e3
rec = defaultdict(dict)   # rec[(step, kernel, call)][slot] = (event, port, tag, cycle)
for step, k, call, r, ev, port, tag, cyc in rows:
    rec[(int(step), int(k), int(call))][int(r)] = (int(ev), int(port), int(tag), int(cyc))
# Since the go-token counter (Iter84 fix): the SLR2 counter's slot 30 (event 18) is its wait for
# the x relay's start token; every other SLR2 stamp is relative to the counter's entry, so
# subtract the wait to make them call-relative (older dumps have no slot 30: nothing changes).
for key, d in rec.items():
    if key[1] == 2 and 30 in d and d[30][0] == 18:
        wait = d[30][3]
        for slot in list(d):
            if slot != 30:
                ev, port, tag, cyc = d[slot]
                d[slot] = (ev, port, tag, cyc - wait)
steps = sorted({s for s, k, c in rec if s > 0})
print(f"steps with records: {len(steps)}; kernel_ms median {med(kernel_ms):.3f} = {tok:,.0f} cycles at 200 MHz")

def val(d, slot):
    return d[slot][3] if slot in d else None

# ---- per-call data
per = defaultdict(lambda: defaultdict(list))   # per[kind][metric] -> values over steps/calls
tags_ok = 0; tags_total = 0
for s in steps:
    for call in range(97):
        c = rec.get((s, 1, call), {}); d = rec.get((s, 2, call), {})
        if not c or not d: continue
        k = kind(call); P = per[k]
        # record lengths
        c_end, d_end = val(c, 31), val(d, 31)
        P["ctrl_len"].append(c_end); P["slr2_len"].append(d_end)
        P["ctrl_seen"].append(c[31][1] if 31 in c else -1); P["slr2_seen"].append(d[31][1] if 31 in d else -1)
        # tag self-test: a "first" stamp's tag is a data byte; count nonzero
        for slot in (2, 4, 6):
            if slot in c: tags_total += 1; tags_ok += c[slot][2] != 0
        # alignment from the two "last" pairs: tx2_last(c) <-> xr_last(s); rx2_last(c) <-> ys_last(s)
        a1 = (val(c, 3), val(d, 1)); a2 = (val(c, 7), val(d, 3))
        if None not in a1 and None not in a2:
            delta = ((a1[0] - a1[1]) + (a2[0] - a2[1])) / 2.0   # SLR2 region start - control region start
            lat = ((a2[0] - a2[1]) - (a1[0] - a1[1])) / 2.0       # one-way link latency
            P["delta"].append(delta); P["link_lat"].append(lat)
            # SLR2's wait for the activation: xr_first on SLR2 = control's tx2_first + lat - delta
            if val(d, 0) is not None: P["slr2_xr_first"].append(val(d, 0))
            if val(c, 2) is not None: P["ctrl_tx2_first"].append(val(c, 2))
            # control-side drain: control end - rx2_last; SLR2 tail: slr2 end - ys_last
            P["ctrl_after_rx2_last"].append(c_end - a2[0]); P["slr2_after_ys_last"].append(d_end - a2[1])
            # weight stream rate once started (per port): weights_done - xr_first over the beats
            for slot, port in ((9, 28), (11, 29), (13, 30), (15, 31)):
                if val(d, slot) is not None and val(d, 0) is not None:
                    P[f"wt_done_{port}"].append(val(d, slot))
            if k == "QKVG":
                for slot, port in ((8, 28), (10, 29), (12, 30), (14, 31)):
                    if val(d, slot) is not None: P[f"pf_done_{port}"].append(val(d, slot))
                for slot, port in ((16, 28), (17, 29), (18, 30), (19, 31)):
                    if val(d, slot) is not None: P[f"wr_done_{port}"].append(val(d, slot))
                for slot in range(20, 30):
                    if val(d, slot) is not None: P[SLR2[slot]].append(val(d, slot))
                for slot in (6, 7, 4, 5):
                    if val(d, slot) is not None: P[SLR2[slot]].append(val(d, slot))
                for slot in range(20, 31):
                    if val(c, slot) is not None: P["c_" + CTRL[slot]].append(val(c, slot))
            else:
                for slot in (20, 21):
                    if val(c, slot) is not None: P["c_" + CTRL[slot]].append(val(c, slot))
        # collector waits per source
        rg = row_groups(call)
        for q in range(4):
            if 16 + q in c:
                P[f"col_sum{q}"].append(c[16 + q][3]); P[f"col_wait{q}"].append(c[16 + q][3] - rg * COL_WORDS[q])
            if 8 + q in c: P[f"col_first{q}"].append(val(c, 8 + q))
            if 12 + q in c: P[f"col_last{q}"].append(val(c, 12 + q))
        for slot in (0, 1, 2, 3, 4, 5, 6, 7):
            if slot in c: P["c_" + CTRL[slot]].append(val(c, slot))
        for slot in (0, 1, 2, 3):
            if slot in d: P["s_" + SLR2[slot]].append(val(d, slot))

print(f"\ntag self-test: {tags_ok}/{tags_total} control 'first' stamps carry a nonzero data byte")
print("\n(a) per call kind, medians, cycles from each region's own start; delta = SLR2 start - control start, lat = link latency")
for k in ("QKVG", "o_proj", "GU", "down", "lm_head"):
    P = per[k]
    if not P["ctrl_len"]: continue
    print(f"  {k:8s} n={len(P['ctrl_len']):4d} ctrl_len={med(P['ctrl_len']):8.0f} slr2_len={med(P['slr2_len']):8.0f} "
          f"seen={med(P['ctrl_seen']):.0f}/{med(P['slr2_seen']):.0f} delta={med(P['delta']):8.1f} lat={med(P['link_lat']):6.1f} "
          f"(lat min..max {min(P['link_lat']):.1f}..{max(P['link_lat']):.1f})")
    print(f"           ctrl: x_bcast {med(P['c_x_bcast_done']):.0f} tx2 {med(P['c_tx2_first']):.0f}..{med(P['c_tx2_last']):.0f} "
          f"w0 {med(P['c_w0_done']):.0f} rx0 {med(P['c_rx0_first']):.0f}..{med(P['c_rx0_last']):.0f} rx2 {med(P['c_rx2_first']):.0f}..{med(P['c_rx2_last']):.0f} "
          f"after_rx2_last {med(P['ctrl_after_rx2_last']):.0f}")
    print(f"           slr2: xr {med(P['s_xr_first']):.0f}..{med(P['s_xr_last']):.0f} ys {med(P['s_ys_first']):.0f}..{med(P['s_ys_last']):.0f} "
          f"wt_done 28..31 {med(P['wt_done_28']):.0f} {med(P['wt_done_29']):.0f} {med(P['wt_done_30']):.0f} {med(P['wt_done_31']):.0f} "
          f"after_ys_last {med(P['slr2_after_ys_last']):.0f}")
    print("           collector: " + " ".join(f"src{q} first {med(P[f'col_first{q}']):.0f} last {med(P[f'col_last{q}']):.0f} sum {med(P[f'col_sum{q}']):.0f} wait {med(P[f'col_wait{q}']):.0f};" for q in range(4)))
    if k == "QKVG":
        print(f"           slr2 qkvg: pf_done {med(P['pf_done_28']):.0f}/{med(P['pf_done_31']):.0f} qkv {med(P['qkv_first']):.0f}..{med(P['qkv_last']):.0f} "
              f"attn {med(P['attn_first']):.0f}..{med(P['attn_last']):.0f} wr_done {med(P['wr_done_28']):.0f}..{med(P['wr_done_31']):.0f}")
        for i in (0, 1):
            print(f"           island {i}: prep_h0 {med(P[f'i{i}_prep_h0']):.0f} read_h0 {med(P[f'i{i}_read_h0']):.0f} read_hl {med(P[f'i{i}_read_hl']):.0f} "
                  f"upd_hl {med(P[f'i{i}_upd_hl']):.0f} upd_end {med(P[f'i{i}_upd_end']):.0f}; per-head read interval {(med(P[f'i{i}_read_hl'])-med(P[f'i{i}_read_h0']))/7:.0f}")
        print(f"           ctrl qkvg: ing_h0 {med(P['c_fill_done|ing_h0']):.0f} ing_h7 {med(P['c_out_done|ing_h7']):.0f} con_h0 {med(P['c_con_h0']):.0f} con_h7 {med(P['c_con_h7']):.0f} "
              f"conv h0 k0/1/2 {med(P['c_conv_h0k0']):.0f}/{med(P['c_conv_h0k1']):.0f}/{med(P['c_conv_h0k2']):.0f} h7k2 {med(P['c_conv_h7k2']):.0f} "
              f"conv_tx_last {med(P['c_conv_tx_last']):.0f} attn_rx {med(P['c_attn_rx_first']):.0f}..{med(P['c_attn_rx_last']):.0f}")
    else:
        print(f"           ctrl store: fill_done {med(P['c_fill_done|ing_h0']):.0f} out_done {med(P['c_out_done|ing_h7']):.0f}")

# ---- (b) token decomposition from the control kernel's view
print("\n(b) token decomposition (medians over steps): control regions, gaps between them (= the control kernel's serial work), SLR2 regions")
gaps = defaultdict(list); sums = []
for s in steps:
    cl = [val(rec.get((s, 1, c), {}), 31) for c in range(97)]
    sl = [val(rec.get((s, 2, c), {}), 31) for c in range(97)]
    if None in cl or None in sl: continue
    sums.append((sum(cl), sum(sl)))
if sums:
    cs, ss = med([a for a, b in sums]), med([b for a, b in sums])
    print(f"  sum of control regions {cs:,.0f} ({100*cs/tok:.1f} % of the token); sum of SLR2 regions {ss:,.0f} ({100*ss/tok:.1f} %); token {tok:,.0f}")
    print(f"  outside the control regions: {tok-cs:,.0f} cycles = the serial work between regions + the top-level drain; outside SLR2's: {tok-ss:,.0f}")
    # the serial work per boundary cannot be stamped directly; infer it from the delta sequence:
    # delta(call+1) - delta(call) = (ctrl region length + ctrl serial gap) - (slr2 region length + slr2 gap) ... printed per kind as a check
    for k in ("QKVG", "o_proj", "GU", "down"):
        print(f"  {k:8s} median ctrl region {med(per[k]['ctrl_len']):8.0f}  slr2 region {med(per[k]['slr2_len']):8.0f}  x24 = {24*med(per[k]['ctrl_len']):,.0f} / {24*med(per[k]['slr2_len']):,.0f}")
    print(f"  lm_head  ctrl {med(per['lm_head']['ctrl_len']):,.0f}  slr2 {med(per['lm_head']['slr2_len']):,.0f}")
    csynth = {"QKVG": 9532, "o_proj": 3224, "GU": 2556, "down": 1800}
    print("  csynth serial work before each call (control kernel, reference): " + ", ".join(f"{k} {v}" for k, v in csynth.items()) + f"; sum x24 = {24*sum(csynth.values()):,}")

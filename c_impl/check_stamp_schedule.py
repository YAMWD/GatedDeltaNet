#!/usr/bin/env python3
"""Iter82 stamp-order gate: in every pipelined-loop module that writes a probe
event (an nbwrite), the event write must be scheduled in the same state as, or
a later state than, every stream/AXIS read of the module -- never earlier
(Iter81's receiver stamp was scheduled one state before the AXIS read).
Usage: check_stamp_schedule.py <solution>/.autopilot/db   -> exit 1 on failure."""
import re, sys, glob, os
db = sys.argv[1]
op_re = re.compile(r"^ST_(\d+) : Operation \d+ \[\d+/\d+\].*?--->\s+\"(.*?)\"(?:\s+\[[^\]]*\])?\s+--->\s+Operation \d+ '(\w+)'")
bad = []; checked = 0
for path in sorted(glob.glob(os.path.join(db, "*.verbose.sched.rpt"))):
    name = os.path.basename(path).replace(".verbose.sched.rpt", "")
    reads, nbw = [], []
    for line in open(path, errors="replace"):
        m = op_re.match(line)
        if not m: continue
        st, text, op = int(m.group(1)), m.group(2), m.group(3)
        if op == "nbwrite" and "ev" in text: nbw.append((st, text[:70]))
        elif op in ("read", "nbread") and re.search(r"_ssdm_op_(Read\.(axis|ap_fifo)|NbRead)", text): reads.append((st, text[:70]))
    if not nbw: continue
    checked += 1
    max_read = max((s for s, _ in reads), default=None)
    min_nbw = min(s for s, _ in nbw)
    pipelined = "_Pipeline_" in name
    ordered = (max_read is None or min_nbw >= max_read)
    # In a pipelined loop module the state numbers are the pipeline stages, so
    # the order is temporal.  In a parent FSM a loop header can carry the
    # post-loop code (executed on exit), so a lower state number is not
    # earlier in time; those are reported for inspection, not failed.
    verdict = "ok" if ordered else ("EARLY" if pipelined else "fsm")
    print(f"{verdict:5s} {name}: reads max ST_{max_read} nbwrite min ST_{min_nbw}")
    if verdict == "EARLY": bad.append(name)
print(f"STAMP_SCHEDULE_{'FAIL' if bad else 'PASS'} modules_with_stamps={checked} early={len(bad)}")
sys.exit(1 if bad else 0)

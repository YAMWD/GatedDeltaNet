# pblock_pe_split.tcl — pre-place TCL invoked by Vivado opt_design.
#
# Splits the 16 ProcessingElement instances of gdn_matmul_systolic across
# SLR0 (PEs 0..7) and SLR1 (PEs 8..15). This relieves the congestion seen
# when the platform's default pinning of gdn_forward_1 to SLR1 forced the
# entire ~1,344-DSP matmul into one SLR (the prior build's failure mode:
# SLR0 at 96% CLB, route_design quit at global congestion level 7).
#
# Each PE has 84 DSPs + 16 RAMB36 + ~11K LUTs; splitting 8/8 gives ~672 DSPs
# per SLR, comfortably below U55C's ~722 DSPs per SLR. The remainder of the
# kernel (recurrent_attention, output_norm, embed, conv1d) is placed by
# Vivado without explicit constraints — with the matmul split off, SLR2's
# previously-empty 84% of CLB area absorbs it naturally.
#
# Invoked via hw.cfg:
#   [vivado]
#   prop=run.impl_1.STEPS.OPT_DESIGN.TCL.PRE={<absolute path>/pblock_pe_split.tcl}
#
# Cell-name pattern matches the HLS-mangled instance names observed in the
# prior build's report_utilization (e.g.
#   level0_i/ulp/gdn_forward_1/inst/grp_gdn_matmul_systolic_fu_1039
#       /grp_run_dataflow_fu_606/ProcessingElement_<N>_U0
# ). add_cells_to_pblock automatically pulls in each PE's c_buf BRAMs and
# pe_k_fu_* pipeline sub-cells.

# Diagnostic: dump every ProcessingElement cell we can see so any future
# name drift is debuggable from the runme.log without re-opening the DCP.
set all_pe_cells [get_cells -hierarchical \
                      -filter {NAME =~ "*ProcessingElement*_U0" && \
                               NAME !~ "*ProcessingElement_*Pipeline*"} -quiet]
puts "pblock_pe_split: found [llength $all_pe_cells] ProcessingElement_*_U0 cells:"
foreach c [lsort $all_pe_cells] {
    puts "  - $c"
}

# ---- Sub-chain A: PEs 0..7 → SLR0 ----
# NOTE: HLS names the FIRST ProcessingElement instance "ProcessingElement_U0"
# (no numeric suffix), and subsequent instances "ProcessingElement_<N>_U0".
# So sub-chain A's 8 cells are {_U0, _1_U0, _2_U0, ..., _7_U0}, not {_0_U0,
# _1_U0, ..., _7_U0}. The unnumbered first-instance pattern is what caused
# the prior build to find only 15 of 16 cells.
create_pblock pblock_pe_lo
resize_pblock pblock_pe_lo -add SLR0

set lo_cells [get_cells -hierarchical \
                  -filter {NAME =~ "*ProcessingElement_U0" || \
                           NAME =~ "*ProcessingElement_1_U0" || \
                           NAME =~ "*ProcessingElement_2_U0" || \
                           NAME =~ "*ProcessingElement_3_U0" || \
                           NAME =~ "*ProcessingElement_4_U0" || \
                           NAME =~ "*ProcessingElement_5_U0" || \
                           NAME =~ "*ProcessingElement_6_U0" || \
                           NAME =~ "*ProcessingElement_7_U0"} -quiet]
puts "pblock_pe_split: pblock_pe_lo got [llength $lo_cells] cells"
if {[llength $lo_cells] > 0} {
    add_cells_to_pblock pblock_pe_lo $lo_cells -quiet
}

# ---- Sub-chain B: PEs 8..15 → SLR1 ----
create_pblock pblock_pe_hi
resize_pblock pblock_pe_hi -add SLR1

set hi_cells [get_cells -hierarchical \
                  -filter {NAME =~ "*ProcessingElement_8_U0" || \
                           NAME =~ "*ProcessingElement_9_U0" || \
                           NAME =~ "*ProcessingElement_10_U0" || \
                           NAME =~ "*ProcessingElement_11_U0" || \
                           NAME =~ "*ProcessingElement_12_U0" || \
                           NAME =~ "*ProcessingElement_13_U0" || \
                           NAME =~ "*ProcessingElement_14_U0" || \
                           NAME =~ "*ProcessingElement_15_U0"} -quiet]
puts "pblock_pe_split: pblock_pe_hi got [llength $hi_cells] cells"
if {[llength $hi_cells] > 0} {
    add_cells_to_pblock pblock_pe_hi $hi_cells -quiet
}

# Sanity: total assigned should be 16. If 0 or partial, the cell-name
# pattern has drifted from the prior build — bail loudly so we don't
# silently waste hours on an unconstrained route.
set total [expr {[llength $lo_cells] + [llength $hi_cells]}]
if {$total != 16} {
    error "pblock_pe_split: expected 16 ProcessingElement cells, found $total \
(lo=[llength $lo_cells] hi=[llength $hi_cells]). See the cell-list dump \
above for the actual names; the HLS-mangled pattern has drifted."
}
puts "pblock_pe_split: assigned 8+8=16 PE cells across SLR0/SLR1"

# =============================================================================
# Non-matmul cell placement
# =============================================================================
# After the previous attempt, route_design still failed with congestion level 7
# in SLR2 (Y=528..591) and the SLR1/SLR2 boundary. Root cause: with the matmul
# pinned to SLR0/SLR1, everything else (recurrent_attention, output_norm,
# conv1d, residuals) collapsed into SLR2 and saturated it locally — the
# offending cell list named gdn_output_norm_and_gate and dozens of fadd_full_dsp
# instances. Distribute them explicitly:
#   - gdn_recurrent_attention (2 MB state, P_K=16 pipeline) → SLR2
#   - gdn_output_norm_and_gate (3 sub-pipelines, ~10K LUTs) → SLR1 (near matmul B)

# ---- Recurrent attention → SLR2 ----
create_pblock pblock_recurrent
resize_pblock pblock_recurrent -add SLR2

set rec_cells [get_cells -hierarchical \
                   -filter {NAME =~ "*grp_gdn_recurrent_attention_fu_*" && \
                            NAME !~ "*grp_gdn_recurrent_attention_*/grp_*"} -quiet]
puts "pblock_pe_split: pblock_recurrent got [llength $rec_cells] cell(s)"
foreach c $rec_cells {
    puts "  - $c"
}
if {[llength $rec_cells] > 0} {
    add_cells_to_pblock pblock_recurrent $rec_cells -quiet
} else {
    puts "pblock_pe_split: WARNING — no grp_gdn_recurrent_attention_fu_* cells matched; \
recurrent attention will be placed wherever Vivado finds room."
}

# ---- Output norm + gate → SLR1 ----
create_pblock pblock_onorm
resize_pblock pblock_onorm -add SLR1

set onorm_cells [get_cells -hierarchical \
                     -filter {NAME =~ "*grp_gdn_output_norm_and_gate_fu_*" && \
                              NAME !~ "*grp_gdn_output_norm_and_gate_*/grp_*"} -quiet]
puts "pblock_pe_split: pblock_onorm got [llength $onorm_cells] cell(s)"
foreach c $onorm_cells {
    puts "  - $c"
}
if {[llength $onorm_cells] > 0} {
    add_cells_to_pblock pblock_onorm $onorm_cells -quiet
} else {
    puts "pblock_pe_split: WARNING — no grp_gdn_output_norm_and_gate_fu_* cells matched; \
output norm will be placed wherever Vivado finds room."
}

# Iter66e pre-place hook: LUT un-pairing only.
#
# The Iter66e source eliminates the cluster clock-enable networks at the HLS
# level (`style=frp` on gemv32_cl_weight_stream), which supersedes Iter66c's
# CE-replication constraints — with frp there may be NO cluster net above the
# 2,000-pin threshold, and iter66c's fail-closed zero-match gate would kill
# the build. This hook therefore chains the production iter54 DMA hook and
# applies only the Iter66d un-pairing lever, which is orthogonal to frp and
# measurably eliminated the intra-site pin-conflict class (Iter66d routed to
# 3 overlaps versus Iter66b's 16).

set iter66e_dir [file dirname [file normalize [info script]]]
source [file join $iter66e_dir apply_iter54_dma_timing.tcl]

set iter66e_family_filter {(NAME =~ *gemv32_four_dots_fu_* || NAME =~ *Pipeline_gemv32_cl_flush* || NAME =~ *m_axi_U/bus_write/fifo_burst*) && IS_PRIMITIVE}

set iter66e_soft [get_cells -hierarchical -quiet -filter \
    "SOFT_HLUTNM != \"\" && $iter66e_family_filter"]
set iter66e_hard [get_cells -hierarchical -quiet -filter \
    "HLUTNM != \"\" && $iter66e_family_filter"]

if {[llength $iter66e_soft] == 0 && [llength $iter66e_hard] == 0} {
    error "iter66e: no paired LUTs found in the targeted families -- hierarchy drift, refusing to continue"
}
if {[llength $iter66e_soft] > 0} {
    set_property SOFT_HLUTNM {} $iter66e_soft
}
if {[llength $iter66e_hard] > 0} {
    set_property HLUTNM {} $iter66e_hard
}

# Report-only: how many high-fanout cluster nets remain with frp active.
set iter66e_ce_nets [get_nets -hierarchical -top_net_of_hierarchical_group -quiet \
    -filter {FLAT_PIN_COUNT > 2000 && TYPE == SIGNAL && NAME =~ *gemv32_cluster2_*}]
puts "GDN_ITER66E_DONE soft_hlutnm_cleared=[llength $iter66e_soft] hlutnm_cleared=[llength $iter66e_hard] residual_ce_nets_over_2000=[llength $iter66e_ce_nets]"

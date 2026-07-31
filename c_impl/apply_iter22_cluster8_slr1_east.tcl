# iter22: minimally relieve iter21's 97.43%-full SLR0 while steering the new
# crossing traffic away from its overloaded central SLL columns.
#
# Iter21's unconstrained placement put clusters 0-2 in SLR2, 4-7 in SLR1, and
# 8-15 mostly in SLR0.  Cluster 8 is therefore the topology boundary: moving it
# to SLR1 changes the contiguous split from ...7|8... to ...8|9... and does not
# introduce an SLR0->SLR1->SLR0 activation-chain round trip.
#
# This is intentionally much narrower than the failed floorplans:
#   * iter12 constrained all 16 clusters to a 6/6/4 split;
#   * iter17 constrained all clusters, their ws/ys FIFOs, and auxiliary stages.
# Here only cluster 8, its two private weight FIFOs, and its already-SLR1 xr[8]
# input FIFO are constrained. Everything else remains available to the placer.
#
# The historical build also imported an iteration-10 binary RQS. That payload
# referred to stale generated hierarchy and is not a portable build input.
# Preserve only its two stable, observable root-level CELL_BLOAT_FACTOR settings
# below as ordinary Tcl; no .rqs file is read by this hook.
#
# Iter21's router estimated the SLR0-SLR1 SLL demand by physical Laguna column
# as:
#   422 1373 1035 1384 1631 1064 1495 1524 1253 657 521 82 91 17 5 4
# The central columns are over capacity, while the east-side columns are nearly
# empty.  ws[16]/ws[17] are tiny BRAM FIFO roots (79 leaf cells each), so place
# just those endpoints in SLR1 clock-region columns X5-X7.  The large cluster is
# assigned only to the full SLR1 and is not hard-packed into that already-busy
# east region.

proc iter22_one_cell {pattern role} {
    set cells [get_cells -hierarchical -quiet -filter "NAME =~ $pattern"]
    if {[llength $cells] != 1} {
        error "iter22: expected one $role for '$pattern', matched [llength $cells]"
    }
    return $cells
}

set hmss_root [iter22_one_cell \
    "level0_i/ulp/hmss_0/inst" "HMSS root"]
set kernel_root [iter22_one_cell \
    "level0_i/ulp/gdn_forward_1/inst" "gdn_forward root"]
set_property CELL_BLOAT_FACTOR low $hmss_root
set_property CELL_BLOAT_FACTOR low $kernel_root

set cluster8 [iter22_one_cell \
    "*/grp_gdn_gemv_fu_*/gemv32_cluster2_8_U0" "cluster 8"]
set ws16 [iter22_one_cell "*/grp_gdn_gemv_fu_*/ws_16_U" "ws[16] FIFO"]
set ws17 [iter22_one_cell "*/grp_gdn_gemv_fu_*/ws_17_U" "ws[17] FIFO"]
set xr8  [iter22_one_cell "*/grp_gdn_gemv_fu_*/xr_8_U"  "xr[8] FIFO"]

# The cluster is the density lever.  Assign the hierarchy root to the full SLR
# and let Vivado choose its X/Y spread and move unrelated logic as needed.
set_property USER_SLR_ASSIGNMENT SLR1 $cluster8
set_property USER_SLR_ASSIGNMENT SLR1 $xr8

# Hard-place only the two BRAM stream endpoints in the underused east side of
# SLR1.  These ranges are the exact X5-X7/Y4-Y7 clock-region site envelope
# measured on the U55C iter21 checkpoint.
set pblock_name pb_iter22_ws16_17_slr1_east
set existing [get_pblocks -quiet $pblock_name]
if {[llength $existing] > 0} {
    delete_pblocks $existing
}
create_pblock $pblock_name
resize_pblock [get_pblocks $pblock_name] -add {
    SLICE_X146Y240:SLICE_X232Y479
    DSP48E2_X20Y90:DSP48E2_X31Y185
    RAMB18_X10Y96:RAMB18_X13Y191
    RAMB36_X10Y48:RAMB36_X13Y95
    URAM288_X3Y64:URAM288_X4Y127
}
set_property CONTAIN_ROUTING 0 [get_pblocks $pblock_name]
set_property USER_SLR_ASSIGNMENT SLR1 $ws16
set_property USER_SLR_ASSIGNMENT SLR1 $ws17
add_cells_to_pblock [get_pblocks $pblock_name] $ws16 -quiet
add_cells_to_pblock [get_pblocks $pblock_name] $ws17 -quiet

puts "GDN_ITER22_CLUSTER cluster8=SLR1 cell=[get_property NAME $cluster8]"
puts "GDN_ITER22_XR xr8=SLR1 cell=[get_property NAME $xr8]"
puts "GDN_ITER22_BLOAT hmss=low kernel=low source=pure_tcl"
puts "GDN_ITER22_EAST_PBLOCK name=$pblock_name roots=[llength [get_cells -quiet -of_objects [get_pblocks $pblock_name]]]"
puts "GDN_ITER22_DONE constrained_roots=4 other_logic=movable"

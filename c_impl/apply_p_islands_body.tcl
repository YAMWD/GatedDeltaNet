# Iter79 step 4 floorplan: one quarter-SLR pblock per recurrent island inside
# the SLR2 kernel.  Measured basis (step 3b, jobs 6008-6010): the island
# closes 5.000 ns at +0.336 ns inside a quarter SLR against +0.147 placed
# freely; its worst path was a fanout-64 BRAM net, which compaction shortens
# (and step 3c halved).  The SLR2 kernel's clusters, adapters, readers, state
# queues and writers stay free in the lower half of SLR2 (Y8-9, beside the
# SLR1 boundary).  Instance names from the gdn_k_slr2 XO's RTL:
#   .../gdn_k_slr2_1/inst/grp_gdn_gemv_part_slr2_fu_*/gdn_recurrent_attention_islands_p_U0/
#       grp_gdn_recurrent_attention_islands_dataflow_p_fu_*/gdn_recurrent_attention_island_{0,1}_U0
# Fail-closed on names.
set p_islands [list \
    [list pb_p_island0 {^.*/gdn_k_slr2_1/.*/gdn_recurrent_attention_island_0_U0$} CLOCKREGION_X0Y10:CLOCKREGION_X3Y11] \
    [list pb_p_island1 {^.*/gdn_k_slr2_1/.*/gdn_recurrent_attention_island_1_U0$} CLOCKREGION_X4Y10:CLOCKREGION_X7Y11]]
foreach spec $p_islands {
    lassign $spec name pattern range
    set cells [get_cells -hierarchical -regexp -quiet $pattern]
    if {[llength $cells] != 1} { error "P_ISLANDS: expected one cell for $name, got [llength $cells]: $cells" }
    set pb [create_pblock $name]
    resize_pblock $pb -add $range
    add_cells_to_pblock $pb $cells
    puts "P_ISLANDS pblock=$name cell=[get_property NAME $cells] range=$range"
}
puts "P_ISLANDS_DONE pblocks=[llength $p_islands]"


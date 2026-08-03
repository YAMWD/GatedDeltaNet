# iter37D: preserve Iter22's proven cluster-8 placement while moving only the
# enlarged 32-lane recurrent hierarchy out of congested SLR1 and into SLR2.
#
# Failed Iter37B placed 17,681 recurrent CLBs, 568 DSPs, and 32 URAMs in SLR1,
# where the block contributed 47% of a level-7 congestion window. SLR2 was only
# 50.36% occupied by CLBs and has sufficient DSP/URAM capacity. Use a full-SLR
# assignment rather than a rectangular pblock so the SSI placer retains freedom
# to spread the hierarchy around the existing GEMV clusters.

set iter37c_script_dir [file dirname [file normalize [info script]]]
source [file join $iter37c_script_dir apply_iter22_cluster8_slr1_east.tcl]

# Match only the HLS function-instance root. Vivado glob '*' also spans '/', so
# the prior NAME filter selected every descendant in the hierarchy.
set iter37c_recurrent [get_cells -hierarchical -quiet -regexp \
    {^.*/grp_gdn_recurrent_attention_fu_[0-9]+$}]
if {[llength $iter37c_recurrent] != 1} {
    error "iter37d: expected one recurrent hierarchy, matched [llength $iter37c_recurrent]"
}

set iter37c_recurrent [lindex $iter37c_recurrent 0]
set_property USER_SLR_ASSIGNMENT SLR2 $iter37c_recurrent

puts "GDN_ITER37D_RECURRENT cell=[get_property NAME $iter37c_recurrent] slr=[get_property USER_SLR_ASSIGNMENT $iter37c_recurrent]"
puts "GDN_ITER37D_DONE recurrent_roots=1 rectangular_pblocks=0"

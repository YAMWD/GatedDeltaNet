# Iter54 pre-place DMA timing repair.
#
# Preserve the proven Iter23/Iter35 DMA fanout fixes, then target the actual
# Iter53 250 MHz critical paths:
#
#   * r15 response FIFO state[0], 529 routed loads;
#   * r15 response FIFO fifoaddr_reg[5], 521 routed loads;
#   * a six-LUT AR-control path that detoured from X3Y1 through X4Y2 and back
#     to X2Y1.
#
# The first two nets are replicated by clock region.  Only the eight primitive
# cells on the measured AR-control path are hard-contained in X2Y1:X3Y1; the
# path-12 HMSS hierarchy and its routing remain otherwise unconstrained.

set iter54_dma_dir [file dirname [file normalize [info script]]]
source [file join $iter54_dma_dir apply_iter35_dma_w15_fifoaddr_fanout.tcl]

proc iter54_dma_one_cell {regexp role} {
    set matches [get_cells -hierarchical -quiet -regexp $regexp]
    if {[llength $matches] != 1} {
        error "iter54 DMA: expected one $role matching '$regexp', matched [llength $matches]"
    }
    return [lindex $matches 0]
}

proc iter54_dma_replicate_q {cell_regexp role} {
    set source [iter54_dma_one_cell $cell_regexp $role]
    set q_pin [get_pins -quiet -of_objects $source -filter {REF_PIN_NAME == Q}]
    if {[llength $q_pin] != 1} {
        error "iter54 DMA: expected one Q pin on $role, matched [llength $q_pin]"
    }
    set net [get_nets -quiet -of_objects $q_pin]
    if {[llength $net] != 1} {
        error "iter54 DMA: expected one net on the $role Q pin, matched [llength $net]"
    }
    set net [lindex $net 0]
    set fanout [llength [get_pins -quiet -of_objects $net -filter {DIRECTION == IN}]]
    if {$fanout < 350 || $fanout > 700} {
        error "iter54 DMA: $role direct fanout $fanout is outside expected pre-place range 350..700"
    }
    set_property MAX_FANOUT_MODE CLOCK_REGION $net
    set_property FORCE_MAX_FANOUT 64 $net
    puts "GDN_ITER54_DMA_NET role=$role cell=[get_property NAME $source] net=[get_property NAME $net] direct_fanout=$fanout mode=[get_property MAX_FANOUT_MODE $net] force_max_fanout=[get_property FORCE_MAX_FANOUT $net]"
}

iter54_dma_replicate_q \
    {^.*/path_12/slice0_12/inst/r15[.]r_multi/triple_slr[.]resp[.]slr_master/common[.]srl_fifo_0/asyncclear_state0_inst$} \
    r15_response_state0
iter54_dma_replicate_q \
    {^.*/path_12/slice0_12/inst/r15[.]r_multi/triple_slr[.]resp[.]slr_master/common[.]srl_fifo_0/fifoaddr_reg\[5\]$} \
    r15_response_fifoaddr5

set iter54_ar_regexps [list \
    {^.*/path_12/switch_2to1_12/inst/s01_entry_pipeline/s01_mmu/inst/areset_reg$} \
    {^.*/path_12/switch_2to1_12/inst/s01_entry_pipeline/s01_mmu/inst/ar_sreg/m_axi_arvalid_INST_0$} \
    {^.*/path_12/switch_2to1_12/inst/s01_entry_pipeline/s01_transaction_regulator/inst/gen_endpoint[.]gen_r_singleorder[.]r_singleorder/gen_id_fifo[.]singleorder_fifo/m_axi_arvalid_INST_0$} \
    {^.*/path_12/switch_2to1_12/inst/s01_entry_pipeline/s01_si_converter/inst/m_axi_arvalid_INST_0$} \
    {^.*/path_12/switch_2to1_12/inst/s01_nodes/s01_ar_node/inst/inst_si_handler/count_r\[5\]_i_1__1$} \
    {^.*/path_12/switch_2to1_12/inst/s01_nodes/s01_ar_node/inst/inst_mi_handler/gen_normal_area[.]inst_fifo_node_payld/gen_xpm_memory_fifo[.]inst_fifo/gen_wr[.]inst_wr_addra_p1/gen_wr[.]full_r_inv_i_3$} \
    {^.*/path_12/switch_2to1_12/inst/s01_nodes/s01_ar_node/inst/inst_mi_handler/gen_normal_area[.]inst_fifo_node_payld/gen_xpm_memory_fifo[.]inst_fifo/gen_wr[.]inst_wr_addra_p1/gen_wr[.]full_r_inv_i_1$} \
    {^.*/path_12/switch_2to1_12/inst/s01_nodes/s01_ar_node/inst/inst_mi_handler/gen_normal_area[.]inst_fifo_node_payld/gen_xpm_memory_fifo[.]inst_fifo/gen_wr[.]full_r_reg_inv$}]

set iter54_ar_cells {}
set iter54_ar_index 0
foreach iter54_ar_regexp $iter54_ar_regexps {
    lappend iter54_ar_cells [iter54_dma_one_cell $iter54_ar_regexp \
        "AR-control path cell $iter54_ar_index"]
    incr iter54_ar_index
}
if {[llength $iter54_ar_cells] != 8} {
    error "iter54 DMA: expected eight unique AR-control path cells, collected [llength $iter54_ar_cells]"
}

# Clock-region objects are named X<n>Y<n>.  The CLOCKREGION_ prefix below is
# resource-range syntax accepted by resize_pblock, not an object-name prefix
# accepted by get_clock_regions.
set iter54_ar_regions [get_clock_regions -quiet {X2Y1 X3Y1}]
if {[llength $iter54_ar_regions] != 2} {
    error "iter54 DMA: expected clock regions X2Y1 and X3Y1, matched [llength $iter54_ar_regions]"
}
puts "GDN_ITER54_AR_REGIONS names=[lsort [get_property NAME $iter54_ar_regions]]"
set iter54_ar_pblock pb_iter54_hmss12_ar_timing
set iter54_ar_existing [get_pblocks -quiet $iter54_ar_pblock]
if {[llength $iter54_ar_existing] > 0} {
    delete_pblocks $iter54_ar_existing
}
create_pblock $iter54_ar_pblock
resize_pblock [get_pblocks $iter54_ar_pblock] -add \
    {CLOCKREGION_X2Y1:CLOCKREGION_X3Y1}
set_property IS_SOFT false [get_pblocks $iter54_ar_pblock]
set_property CONTAIN_ROUTING false [get_pblocks $iter54_ar_pblock]
add_cells_to_pblock [get_pblocks $iter54_ar_pblock] $iter54_ar_cells

puts "GDN_ITER54_AR_PBLOCK name=$iter54_ar_pblock cells=[llength $iter54_ar_cells] ranges=[get_property GRID_RANGES [get_pblocks $iter54_ar_pblock]] soft=[get_property IS_SOFT [get_pblocks $iter54_ar_pblock]] contain_routing=[get_property CONTAIN_ROUTING [get_pblocks $iter54_ar_pblock]]"
puts "GDN_ITER54_DMA_DONE iter35_preserved=1 r15_state0_fanout=64 r15_fifoaddr5_fanout=64 ar_path_localized=1"

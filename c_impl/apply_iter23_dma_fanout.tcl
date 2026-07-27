# iter23: close the fixed 250 MHz dma_ip_axi_aclk_1 timing failure seen after
# iter22 routed legally.
#
# The iter22 worst DMA path is almost entirely routing:
#   source   .../common.srl_fifo_0/asyncclear_state1_inst/Q
#   net      .../common.srl_fifo_0/state[1]
#   fanout   524
#   route    3.945 ns of a 4.113 ns data path
#   WNS      -0.307 ns
#
# Run this hook immediately before place_design so the placer sees localized
# replicas. MAX_FANOUT_MODE partitions loads by clock region; the explicit
# limit asks for roughly nine or more copies instead of leaving the original
# driver to span the SLR2 FIFO footprint. Keep the match strict because this
# is fixed platform/HMSS logic, not the HLS kernel.

# Vivado treats periods in generated hierarchical names as pattern syntax, so
# neither a full-name pattern nor a long NAME filter is reliable here. Search
# by the stable leaf suffix, then use Tcl's regexp engine (not Vivado's object
# pattern parser) to identify the r15 read-response FDCE.
set iter23_source_regexp \
    {r15[.]r_multi/triple_slr[.]resp[.]slr_master/common[.]srl_fifo_0/asyncclear_state1_inst$}
set iter23_source_candidates \
    [get_cells -hierarchical -quiet -filter {NAME =~ *asyncclear_state1_inst*}]
set iter23_source_cells {}
foreach iter23_candidate $iter23_source_candidates {
    set iter23_candidate_name [string trim [get_property NAME $iter23_candidate]]
    if {[regexp $iter23_source_regexp $iter23_candidate_name]} {
        lappend iter23_source_cells $iter23_candidate
    }
}
if {[llength $iter23_source_cells] != 1} {
    puts "GDN_ITER23_DMA_SOURCE_DIAGNOSTIC string_matches=[llength $iter23_source_cells] suffix_matches=[llength $iter23_source_candidates]"
    foreach iter23_candidate $iter23_source_candidates {
        puts "GDN_ITER23_DMA_SOURCE_CANDIDATE name=[get_property NAME $iter23_candidate]"
    }
    error "iter23: expected one DMA state[1] source matching '$iter23_source_regexp', matched [llength $iter23_source_cells]"
}
set iter23_source [lindex $iter23_source_cells 0]

set iter23_q_pins \
    [get_pins -quiet -of_objects $iter23_source -filter {REF_PIN_NAME == Q}]
if {[llength $iter23_q_pins] != 1} {
    error "iter23: expected one Q pin on '[get_property NAME $iter23_source]', matched [llength $iter23_q_pins]"
}
set iter23_q_pin [lindex $iter23_q_pins 0]

set iter23_nets [get_nets -quiet -of_objects $iter23_q_pin]
if {[llength $iter23_nets] != 1} {
    error "iter23: expected one net on '[get_property NAME $iter23_q_pin]', matched [llength $iter23_nets]"
}
set iter23_state_net [lindex $iter23_nets 0]

set iter23_load_pins \
    [get_pins -quiet -of_objects $iter23_state_net -filter {DIRECTION == IN}]
set iter23_original_fanout [llength $iter23_load_pins]
if {$iter23_original_fanout < 64} {
    error "iter23: DMA target fanout unexpectedly small: $iter23_original_fanout"
}

set_property MAX_FANOUT_MODE CLOCK_REGION $iter23_state_net
set_property FORCE_MAX_FANOUT 64 $iter23_state_net

puts "GDN_ITER23_DMA_SOURCE cell=[get_property NAME $iter23_source]"
puts "GDN_ITER23_DMA_NET net=[get_property NAME $iter23_state_net] original_fanout=$iter23_original_fanout"
puts "GDN_ITER23_DMA_PROPERTIES max_fanout_mode=[get_property MAX_FANOUT_MODE $iter23_state_net] force_max_fanout=[get_property FORCE_MAX_FANOUT $iter23_state_net]"
puts "GDN_ITER23_DMA_DONE"

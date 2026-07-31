# iter35: corrected pre-place form of the iter34 fixed-DMA repair.
#
# Iter33 final routed timing identified fifoaddr_reg[2] and [4] in the w15
# forward-path SRL FIFO as the only remaining 250 MHz DMA setup sources. The
# routed high-fanout report counts 1159 and 1157 timing loads, while the
# pre-place netlist exposes 582 direct input pins on fifoaddr_reg[2]. Iter34
# stopped intentionally because its 1000..1300 direct-pin guard used the
# post-route count. This hook keeps the exact hierarchy match and accepts the
# observed pre-place representation.

set iter35_script_dir [file dirname [file normalize [info script]]]
source [file join $iter35_script_dir apply_iter23_dma_fanout.tcl]

set iter35_source_regexp \
    {path_12/slice0_12/inst/w15[.]w_multi/triple_slr[.]fwd[.]slr_slave/common[.]srl_fifo_0/fifoaddr_reg\[([24])\]$}
set iter35_source_candidates \
    [get_cells -hierarchical -quiet -filter {NAME =~ *fifoaddr_reg*}]
array set iter35_sources {}
foreach iter35_candidate $iter35_source_candidates {
    set iter35_candidate_name [string trim [get_property NAME $iter35_candidate]]
    if {[regexp $iter35_source_regexp $iter35_candidate_name -> iter35_bit]} {
        if {[info exists iter35_sources($iter35_bit)]} {
            error "iter35: duplicate w15 fifoaddr_reg[$iter35_bit] target"
        }
        set iter35_sources($iter35_bit) $iter35_candidate
    }
}

foreach iter35_bit {2 4} {
    if {![info exists iter35_sources($iter35_bit)]} {
        puts "GDN_ITER35_DMA_SOURCE_DIAGNOSTIC bit=$iter35_bit candidates=[llength $iter35_source_candidates]"
        error "iter35: expected one w15 fifoaddr_reg[$iter35_bit] target matching '$iter35_source_regexp'"
    }

    set iter35_source $iter35_sources($iter35_bit)
    set iter35_q_pins \
        [get_pins -quiet -of_objects $iter35_source -filter {REF_PIN_NAME == Q}]
    if {[llength $iter35_q_pins] != 1} {
        error "iter35: expected one Q pin on '[get_property NAME $iter35_source]', matched [llength $iter35_q_pins]"
    }
    set iter35_q_pin [lindex $iter35_q_pins 0]

    set iter35_nets [get_nets -quiet -of_objects $iter35_q_pin]
    if {[llength $iter35_nets] != 1} {
        error "iter35: expected one net on '[get_property NAME $iter35_q_pin]', matched [llength $iter35_nets]"
    }
    set iter35_net [lindex $iter35_nets 0]

    set iter35_load_pins \
        [get_pins -quiet -of_objects $iter35_net -filter {DIRECTION == IN}]
    set iter35_original_fanout [llength $iter35_load_pins]
    if {$iter35_original_fanout < 400 || $iter35_original_fanout > 700} {
        error "iter35: w15 fifoaddr_reg[$iter35_bit] direct fanout $iter35_original_fanout is outside expected pre-place range 400..700"
    }

    set_property MAX_FANOUT_MODE CLOCK_REGION $iter35_net
    set_property FORCE_MAX_FANOUT 64 $iter35_net

    puts "GDN_ITER35_DMA_SOURCE bit=$iter35_bit cell=[get_property NAME $iter35_source]"
    puts "GDN_ITER35_DMA_NET bit=$iter35_bit net=[get_property NAME $iter35_net] direct_fanout=$iter35_original_fanout"
    puts "GDN_ITER35_DMA_PROPERTIES bit=$iter35_bit max_fanout_mode=[get_property MAX_FANOUT_MODE $iter35_net] force_max_fanout=[get_property FORCE_MAX_FANOUT $iter35_net]"
}

puts "GDN_ITER35_DMA_DONE"

# Capture normal evidence first, then reject a miss before clock scaling.
source [file join [file dirname [file normalize [info script]]] report_final_qor.tcl]
set fp [open [file join $gdn_final_qor_dir exact_clock_gate.tsv] w]
puts $fp "clock\tperiod_ns\tsetup_wns_ns\thold_whs_ns"
set failed {}
foreach {name expected} {clk_kernel_00_unbuffered_net 6.667 dma_ip_axi_aclk_1 4.000 hbm_aclk 2.222} {
    set clk [get_clocks -quiet $name]
    if {[llength $clk] != 1} {error "ITER75D: missing required clock $name"}
    set period [get_property PERIOD $clk]
    if {abs($period-$expected)>0.002} {error "ITER75D: wrong $name period $period"}
    set setup [get_timing_paths -quiet -from $clk -to $clk -delay_type max -max_paths 1]
    set hold [get_timing_paths -quiet -from $clk -to $clk -delay_type min -max_paths 1]
    if {![llength $setup] || ![llength $hold]} {error "ITER75D: missing $name timing paths"}
    set wns [get_property SLACK $setup]
    set whs [get_property SLACK $hold]
    puts $fp "$name\t$period\t$wns\t$whs"
    puts "ITER75D_CLOCK clock=$name period=$period setup=$wns hold=$whs"
    if {$wns<0 || $whs<0} {lappend failed $name}
}
close $fp
if {[llength $failed]} {error "ITER75D: exact-clock timing failed: $failed; scaling is not accepted"}
puts "ITER75D_EXACT_CLOCK_TIMING_PASS"

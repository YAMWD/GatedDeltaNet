# Capture normal evidence first, then reject a miss before clock scaling.
source [file join [file dirname [file normalize [info script]]] report_final_qor.tcl]
set fp [open [file join $gdn_final_qor_dir exact_clock_gate.tsv] w]
puts $fp "clock\tperiod_ns\tsetup_wns_ns\thold_whs_ns"
set failed {}
# Kernel period from GDN_KERNEL_TARGET_MHZ (default 150 -> 6.667); the DMA and
# HBM periods belong to the shell and are fixed.
set gate_target_mhz [expr {[info exists ::env(GDN_KERNEL_TARGET_MHZ)] ? double($::env(GDN_KERNEL_TARGET_MHZ)) : 150.0}]
set gate_kernel_ns [expr {double(round(1000000.0 / $gate_target_mhz)) / 1000.0}]
foreach {name expected} [list clk_kernel_00_unbuffered_net $gate_kernel_ns dma_ip_axi_aclk_1 4.000 hbm_aclk 2.222] {
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
    # Iter79 partition: the shell's SLR-crossing pipes at hbm_aclk have no
    # margin for masters outside SLR0 (P2/P3/P4: -0.028/-0.288/-0.018).  With
    # GDN_HBM_MIN_MHZ set, hbm_aclk may miss 2.222 ns by what a scaled clock at
    # that frequency would absorb (vpl's auto-scaling then programs it); the
    # kernel and DMA clocks stay exact.  Unset, the gate is the production one.
    set tol 0.0
    if {$name eq "hbm_aclk" && [info exists ::env(GDN_HBM_MIN_MHZ)]} {
        set tol [expr {1000.0 / double($::env(GDN_HBM_MIN_MHZ)) - $expected}]
        puts "ITER75D_CLOCK hbm_aclk tolerance=[format %.3f $tol] ns (GDN_HBM_MIN_MHZ=$::env(GDN_HBM_MIN_MHZ))"
    }
    if {$wns < -$tol || $whs<0} {lappend failed $name}
}
close $fp
if {[llength $failed]} {error "ITER75D: exact-clock timing failed: $failed; scaling is not accepted (kernel/DMA exact; hbm_aclk within GDN_HBM_MIN_MHZ if set)"}
puts "ITER75D_EXACT_CLOCK_TIMING_PASS"

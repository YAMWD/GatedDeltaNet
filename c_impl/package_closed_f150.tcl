# Package the timing-closed 150 MHz checkpoint into a partial bitstream.
# Fail closed: re-verify the closure on a fresh open before writing anything,
# so a mislabelled checkpoint can never become an image.
set dcp $::env(PK_DCP); set out $::env(PK_OUT); file mkdir $out
proc note {m} { puts "ITER75F_PKG [clock format [clock seconds] -format %H:%M:%S] $m"; flush stdout }
if {[catch {open_checkpoint $dcp} e]} { note "ABORT open: $e"; exit 3 }
note "opened $dcp"
# 1. route must be legal AND complete: zero errors and every routable net routed
report_route_status -file $out/route_status.rpt
set fh [open $out/route_status.rpt r]; set route [read $fh]; close $fh
foreach {key label} {total {routable nets} fully {fully routed nets} errors {nets with routing errors}} {
    if {![regexp [format {# of %s\.+\s*:\s*([0-9]+)} $label] $route -> n]} { note "ABORT: route-status field missing: $label"; exit 3 }
    set counts($key) $n
}
note "routable=$counts(total) fully_routed=$counts(fully) routing_errors=$counts(errors)"
if {$counts(errors) != 0 || $counts(total) == 0 || $counts(total) != $counts(fully)} { note "ABORT: route is not legal and complete"; exit 3 }
# 2. all three clocks must exist at their exact periods, have both setup and hold
#    evidence, and meet both. A missing path is a failure, not a pass.
set fh [open $out/clock_slacks.tsv w]; puts $fh "clock\tperiod\tsetup\thold"
set bad 0
foreach {cn expected} {clk_kernel_00_unbuffered_net 6.667 dma_ip_axi_aclk_1 4.000 hbm_aclk 2.222} {
    set c [get_clocks -quiet $cn]
    if {[llength $c] != 1} { note "ABORT: clock $cn not found"; exit 3 }
    set per [get_property PERIOD $c]
    if {abs($per - $expected) > 0.002} { note "ABORT: $cn period $per is not $expected"; exit 3 }
    set sp [get_timing_paths -quiet -to $c -delay_type max -max_paths 1]
    set hp [get_timing_paths -quiet -to $c -delay_type min -max_paths 1]
    if {![llength $sp] || ![llength $hp]} { note "ABORT: $cn has no setup/hold timing path to evaluate"; exit 3 }
    set ss [get_property SLACK [lindex $sp 0]]
    set hs [get_property SLACK [lindex $hp 0]]
    puts $fh "$cn\t$per\t$ss\t$hs"; note "CLOCK $cn period=$per setup=$ss hold=$hs"
    if {$ss < 0 || $hs < 0} { incr bad }
}
close $fh
if {$bad} { note "ABORT: $bad clock(s) still failing -- refusing to package"; exit 3 }
note "CLOSURE_REVERIFIED: three clocks pass at 6.667 / 4.000 / 2.222 ns, route legal and complete"
# 3. find the reconfigurable partition, do not assume its name
set rp ""
foreach c [get_cells -hierarchical -quiet -filter {HD.RECONFIGURABLE == 1}] { lappend rp [get_property NAME $c] }
note "reconfigurable cells: $rp"
if {[llength $rp] != 1} { note "ABORT: expected exactly one reconfigurable partition"; exit 3 }
# 4. write the partial bitstream
if {[catch {write_bitstream -force -cell [get_cells [lindex $rp 0]] $out/ulp_partial.bit} e]} { note "ABORT write_bitstream: $e"; exit 3 }
note "BITFILE [file size $out/ulp_partial.bit] bytes"
note "PACKAGE_DONE"
exit 0

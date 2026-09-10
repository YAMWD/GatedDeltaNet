# Package the timing-closed 150 MHz checkpoint into a partial bitstream.
# Fail closed: re-verify the closure on a fresh open before writing anything,
# so a mislabelled checkpoint can never become an image.
set dcp $::env(PK_DCP); set out $::env(PK_OUT); file mkdir $out
proc note {m} { puts "ITER75F_PKG [clock format [clock seconds] -format %H:%M:%S] $m"; flush stdout }
if {[catch {open_checkpoint $dcp} e]} { note "ABORT open: $e"; exit 3 }
note "opened $dcp"
# 1. route must be legal
report_route_status -file $out/route_status.rpt
set errs -1; set fh [open $out/route_status.rpt r]
while {[gets $fh l] >= 0} { if {[regexp {nets with routing errors\.+ :\s+(\d+)} $l -> n]} {set errs $n} }
close $fh
note "routing_errors=$errs"
if {$errs != 0} { note "ABORT: not legally routed"; exit 3 }
# 2. all three clocks must meet setup AND hold at the real constraint
set fh [open $out/clock_slacks.tsv w]; puts $fh "clock\tperiod\tsetup\thold"
set bad 0
foreach cn {clk_kernel_00_unbuffered_net dma_ip_axi_aclk_1 hbm_aclk} {
    set c [get_clocks -quiet $cn]
    if {[llength $c] != 1} { note "ABORT: clock $cn not found"; exit 3 }
    set per [get_property PERIOD $c]
    set sp [get_timing_paths -quiet -to $c -delay_type max -max_paths 1]
    set hp [get_timing_paths -quiet -to $c -delay_type min -max_paths 1]
    set ss [expr {[llength $sp] ? [get_property SLACK [lindex $sp 0]] : 99}]
    set hs [expr {[llength $hp] ? [get_property SLACK [lindex $hp 0]] : 99}]
    puts $fh "$cn\t$per\t$ss\t$hs"; note "CLOCK $cn period=$per setup=$ss hold=$hs"
    if {$ss < 0 || $hs < 0} { incr bad }
}
close $fh
if {$bad} { note "ABORT: $bad clock(s) still failing -- refusing to package"; exit 3 }
if {abs([get_property PERIOD [get_clocks clk_kernel_00_unbuffered_net]] - 6.667) > 0.002} { note "ABORT: kernel period is not 6.667"; exit 3 }
note "CLOSURE_REVERIFIED: three clocks pass at 6.667 ns, route legal"
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

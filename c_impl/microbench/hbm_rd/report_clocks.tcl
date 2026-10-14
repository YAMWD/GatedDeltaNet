# ROUTE_DESIGN.TCL.POST: per-clock setup/hold slack of the routed design (no gate: this is a
# diagnostic image; vpl's auto-scaling, if any, is read back from the xclbin and the card).
set fp [open [file join [pwd] hbm_rd_clocks.tsv] w]
puts $fp "clock\tperiod_ns\tsetup_wns_ns\thold_whs_ns\tfailing_endpoints"
foreach name {clk_kernel_00_unbuffered_net dma_ip_axi_aclk_1 hbm_aclk} {
    set c [get_clocks -quiet $name]
    if {[llength $c] != 1} { puts $fp "$name\tmissing\t\t\t"; continue }
    set s [get_timing_paths -quiet -from $c -to $c -setup -max_paths 1]
    set h [get_timing_paths -quiet -from $c -to $c -hold -max_paths 1]
    set nf [llength [get_timing_paths -quiet -from $c -to $c -setup -max_paths 100000 -slack_lesser_than 0]]
    puts $fp "$name\t[get_property PERIOD $c]\t[get_property SLACK $s]\t[get_property SLACK $h]\t$nf"
}
close $fp

# OPT_DESIGN.TCL.PRE: force the kernel clock to GDN_KERNEL_TARGET_MHZ (Vitis 2024.2 writes a 1:1
# MMCM ratio into _user_impl_clk.xdc regardless of --kernel_frequency; see
# c_impl/apply_iter69_kernel_clock_f150.tcl). Fails closed if the period does not take.
set target_mhz [expr {[info exists ::env(GDN_KERNEL_TARGET_MHZ)] ? double($::env(GDN_KERNEL_TARGET_MHZ)) : 300.0}]
set mmcm level0_i/ulp/ulp_ucs/inst/aclk_kernel_00_hierarchy/clkwiz_aclk_kernel_00/inst/CLK_CORE_DRP_I/clk_inst/mmcme4_adv_inst
set clk [get_clocks -quiet clk_kernel_00_unbuffered_net]
if {[llength $clk] != 1} { error "kernel clock: expected one clk_kernel_00_unbuffered_net, found [llength $clk]" }
set in [get_pins -quiet $mmcm/CLKIN1]; set out [get_pins -quiet $mmcm/CLKOUT0]
if {[llength $in] != 1 || [llength $out] != 1} { error "kernel clock: MMCM pins not found under $mmcm" }
set in_mhz [expr {1000.0 / [get_property PERIOD [get_clocks -of_objects $in]]}]
set found 0
for {set div 1} {$div <= 16 && !$found} {incr div} {
    set mult [expr {round($target_mhz * $div / $in_mhz)}]
    if {abs($in_mhz * $mult / $div - $target_mhz) < 0.01} { set found 1; break }
}
if {!$found} { error "kernel clock: no integer ratio from $in_mhz MHz to $target_mhz MHz" }
puts "kernel clock: before override period=[get_property PERIOD $clk] ns; input $in_mhz MHz; x$mult/$div for $target_mhz MHz"
create_generated_clock -name clk_kernel_00_unbuffered_net -source $in -multiply_by $mult -divide_by $div $out
set after [get_property PERIOD [get_clocks clk_kernel_00_unbuffered_net]]
set want [expr {1000.0 / $target_mhz}]
puts "kernel clock: after override period=$after ns (want $want ns)"
if {abs($after - $want) > 0.002} { error "kernel clock: override did not take ($after ns, wanted $want ns)" }

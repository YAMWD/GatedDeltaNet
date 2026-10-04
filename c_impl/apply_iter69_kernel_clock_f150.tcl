# Iter69: put the requested kernel frequency back into the implementation
# timing constraint.
#
# Vitis 2024.2's vpl writes the kernel clock as a generated clock on the ucs
# MMCM output (`_user_impl_clk.xdc`, ocl_util.tcl write_user_impl_clock_
# constraint).  In 2022.2 the ratio came out right (100 MHz request ->
# `-divide_by 12 -multiply_by 12` from the 100 MHz free-running input); in
# 2024.2 the same Tcl obtained `-divide_by 1 -multiply_by 1` for BOTH the
# 100 MHz (Iter67c, post_place.dcp late.xdc:65475) and the 150 MHz (Iter68G,
# late.xdc:65475) requests.  100 MHz happened to be the 1:1 answer, so every
# 2024.2 build so far was implemented at 10.000 ns; the Iter68G "150 MHz"
# request was implemented at 10.000 ns and then AUTO-FREQ-SCALING-04 scaled the
# programmed DATA_CLK to 101.7 MHz.  The synthesis-side constraint
# (`create_clock -period 6.667` in ulp_ooc_copy.xdc) was correct in both.
#
# This hook runs as OPT_DESIGN.TCL.PRE with the full design open and
# re-creates the generated clock with the ratio for the requested frequency,
# then fails closed unless the clock period is what was asked for.  The MMCM
# multiply/divide here are a timing ratio only; XRT programs the real MMCM by
# DRP from CLOCK_FREQ_TOPOLOGY at xclbin load.
#
# Requested: 150 MHz from the 100 MHz `io_clk_freerun_00_clk_p` = x3/2.

# Target from GDN_KERNEL_TARGET_MHZ (hw_build.slurm exports LINK_FREQ there);
# default 150. The multiply/divide pair is the smallest integer ratio from the
# MMCM input (100 MHz), so 150 -> x3/2 exactly as before, 200 -> x2/1.
set iter69_target_mhz [expr {[info exists ::env(GDN_KERNEL_TARGET_MHZ)] ? double($::env(GDN_KERNEL_TARGET_MHZ)) : 150.0}]
set iter69_mult 0
set iter69_div 0

set iter69_mmcm level0_i/ulp/ulp_ucs/inst/aclk_kernel_00_hierarchy/clkwiz_aclk_kernel_00/inst/CLK_CORE_DRP_I/clk_inst/mmcme4_adv_inst
set iter69_clk [get_clocks -quiet clk_kernel_00_unbuffered_net]
if {[llength $iter69_clk] != 1} {
    error "iter69 kernel clock: expected one clock named clk_kernel_00_unbuffered_net, found [llength $iter69_clk]"
}
set iter69_before [get_property PERIOD $iter69_clk]
set iter69_in [get_pins -quiet $iter69_mmcm/CLKIN1]
set iter69_out [get_pins -quiet $iter69_mmcm/CLKOUT0]
if {[llength $iter69_in] != 1 || [llength $iter69_out] != 1} {
    error "iter69 kernel clock: MMCM pins not found under $iter69_mmcm"
}
set iter69_in_period [get_property PERIOD [get_clocks -of_objects $iter69_in]]
set iter69_in_mhz [expr {1000.0 / $iter69_in_period}]
for {set d 1} {$d <= 16} {incr d} {
    set m [expr {round($iter69_target_mhz * $d / $iter69_in_mhz)}]
    if {abs($iter69_in_mhz * $m / $d - $iter69_target_mhz) < 0.01} { set iter69_mult $m; set iter69_div $d; break }
}
if {$iter69_mult == 0} { error "iter69 kernel clock: no integer MMCM ratio from $iter69_in_mhz MHz to $iter69_target_mhz MHz" }
puts "iter69 kernel clock: before override period=$iter69_before ns, MMCM input period=$iter69_in_period ns, target $iter69_target_mhz MHz = x$iter69_mult/$iter69_div"

create_generated_clock -name clk_kernel_00_unbuffered_net \
    -source $iter69_in -multiply_by $iter69_mult -divide_by $iter69_div $iter69_out

set iter69_clk [get_clocks clk_kernel_00_unbuffered_net]
set iter69_after [get_property PERIOD $iter69_clk]
set iter69_want [expr {1000.0 / $iter69_target_mhz}]
puts "iter69 kernel clock: after override period=$iter69_after ns (want $iter69_want ns, x$iter69_mult/$iter69_div)"
if {abs($iter69_after - $iter69_want) > 0.002} {
    error "iter69 kernel clock: override did not take, period is $iter69_after ns, wanted $iter69_want ns"
}
set iter69_rpt_dir [file join [file dirname [file normalize [info script]]] diagnostics placement_reports]
file mkdir $iter69_rpt_dir
report_clocks -file [file join $iter69_rpt_dir iter69_clocks_after_override.rpt]

# Then the unchanged Iter66e floorplan.
set iter69_dir [file dirname [file normalize [info script]]]
# Iter77 S2: a caller (the 200 MHz hook) may substitute the floorplan script
# for a source whose instance names differ; the production default is unchanged.
if {![info exists gdn_islands_script]} {
    set gdn_islands_script [file join $iter69_dir apply_f150_physical_islands.tcl]
}
source $gdn_islands_script

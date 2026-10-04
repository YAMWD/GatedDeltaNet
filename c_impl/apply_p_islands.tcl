# Iter79 step 4 (OPT_DESIGN.PRE): the true kernel clock, then the partition's
# own floorplan.  The Iter69 clock hook sources a floorplan script after the
# clock override -- the production Iter56/66e islands by default, which look
# for the monolithic hierarchy and failed closed on the partitioned netlist
# (link P1, job 6023, 2 h in) -- so this hook substitutes its own body
# through the hook's gdn_islands_script hand-off.
set p_dir [file dirname [file normalize [info script]]]
set gdn_islands_script [file join $p_dir apply_p_islands_body.tcl]
source [file join $p_dir apply_iter69_kernel_clock_f150.tcl]

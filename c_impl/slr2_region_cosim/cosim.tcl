open_project -reset GDN_slr2_region_cosim
set_top gdn_slr2_region_cosim_top
add_files -cflags "-DGDN_SLR2_REGION_COSIM -ffp-contract=off" gdn_model.cpp
add_files gdn_model.h
add_files -tb slr2_region_cosim_tb.cpp -cflags "-ffp-contract=off -I."
open_solution -reset solution1 -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 5.0 -name default
source ./hls_gdn_forward.tcl
csynth_design
cosim_design -rtl verilog -trace_level none
exit

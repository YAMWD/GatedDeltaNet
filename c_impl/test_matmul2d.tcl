open_project GDN_matmul2d
set_top gdn_matmul2d_top
add_files gdn_model.cpp
add_files gdn_model.h
add_files -tb gdn_matmul2d_test.cpp
open_solution -reset "solution1" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_clock 10
# csynth only — parity is checked natively via the Makefile gdn_matmul2d_test
# target (csim of the 2048^3 naive golden is slow). Latency is read from the
# csynth report's top-level (gdn_matmul2d_top) row.
csynth_design
exit

open_project GDN_matmul1_synth
set_top gdn_matmul_top
add_files gdn_model.cpp
add_files gdn_model.h
add_files -tb gdn_matmul_test.cpp
open_solution -reset "solution1" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_clock 10
# csynth only (parity verified natively via Makefile gdn_matmul_test).
csynth_design

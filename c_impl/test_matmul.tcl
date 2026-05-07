open_project GDN_matmul
set_top gdn_matmul_top
add_files gdn_model.c
add_files gdn_model.h
add_files -tb gdn_matmul_test.c
open_solution -reset "solution1" -flow_target vitis
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_clock 10
# csim_design args: num_rows in_dim out_dim seed
# Full-scale 2048 x 2048 x 2048 — matches the dominant matmul shape in
# GDN-1.3B (Q/K/V/gate/output_proj projections). For faster iteration
# during early bring-up, drop to e.g. {64 64 64 42} or {256 256 256 7}.
csim_design -argv {2048 2048 2048 42}
csynth_design
# cosim_design -argv {2048 2048 2048 42}
# export_design -flow syn -rtl verilog -format ip_catalog

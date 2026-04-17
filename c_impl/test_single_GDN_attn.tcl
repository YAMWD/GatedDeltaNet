open_project GDN_single_attn
set_top gdn_attn_forward
add_files gdn_model.c
add_files gdn_model.h
add_files -tb gdn_attn_test.c
add_files -tb artifacts
add_files -tb fixtures_block
open_solution -reset "solution2" -flow_target vitis
set_part {xcvu11p-flga2577-1-e}
create_clock -period 10 -name default
config_export -format ip_catalog -rtl verilog -vivado_clock 10
#source "./GDN_single_attn/solution2/directives.tcl"
csim_design -argv {artifacts/gdn-1.3b-f32.gdnw fixtures_block/block0_attn.gdnblk}
csynth_design
cosim_design -argv {artifacts/gdn-1.3b-f32.gdnw fixtures_block/block0_attn.gdnblk}
#export_design -flow syn -rtl verilog -format ip_catalog
# Keep instrumentation and reset distribution from becoming global routing hubs.
config_interface -m_axi_alignment_byte_size 64 -m_axi_latency 64 -m_axi_max_widen_bitwidth 512
config_rtl -kernel_profile=false
config_rtl -reset control
config_rtl -register_reset_num 0

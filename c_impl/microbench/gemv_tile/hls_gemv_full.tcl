# Keep instrumentation and reset distribution from becoming global routing hubs.
config_rtl -kernel_profile=false
config_rtl -reset control
config_rtl -register_reset_num 0

SHELL         := /bin/bash
.SHELLFLAGS   := -o pipefail -c
VITIS_VERSION ?= 2024.2
XILINX_VITIS  ?= /tools/Xilinx/Vitis/$(VITIS_VERSION)
XILINX_XRT    ?= /opt/xilinx/xrt
VPP           ?= $(XILINX_VITIS)/bin/v++
PLATFORM      ?= xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET        ?= hw
FREQ          ?= 200
JOBS          ?= 8
SCDEPTH       ?=
BUILD_DIR     := build.$(TARGET).slr_stream.f$(FREQ)$(if $(SCDEPTH),.sc$(SCDEPTH),)
CLOCK_TCL     := $(abspath ../gemv_tile/apply_kernel_clock_target.tcl)
VPP_FLAGS     := -t $(TARGET) --platform $(PLATFORM) --save-temps --optimize 2 --hls.jobs $(JOBS) --kernel_frequency $(FREQ)
SC            := $(if $(SCDEPTH),:$(SCDEPTH),)
VPP_LDFLAGS   := --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
VPP_LDFLAGS   += --connectivity.nk slr_src:3:slr_src_1.slr_src_2.slr_src_3
VPP_LDFLAGS   += --connectivity.nk slr_sink:3:slr_sink_1.slr_sink_2.slr_sink_3
VPP_LDFLAGS   += --connectivity.sc slr_src_1.out:slr_sink_1.in$(SC) --connectivity.sc slr_src_2.out:slr_sink_2.in$(SC) --connectivity.sc slr_src_3.out:slr_sink_3.in$(SC)
VPP_LDFLAGS   += --connectivity.slr slr_src_1:SLR0 --connectivity.slr slr_sink_1:SLR1
VPP_LDFLAGS   += --connectivity.slr slr_src_2:SLR1 --connectivity.slr slr_sink_2:SLR2
VPP_LDFLAGS   += --connectivity.slr slr_src_3:SLR0 --connectivity.slr slr_sink_3:SLR2
VPP_LDFLAGS   += --connectivity.sp slr_sink_1.out:HBM[0] --connectivity.sp slr_sink_2.out:HBM[1] --connectivity.sp slr_sink_3.out:HBM[2]
VPP_LDFLAGS   += --vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$(CLOCK_TCL)
HOST_INC      := -I$(XILINX_VITIS)/include -I$(XILINX_XRT)/include
HOST_LIB      := -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib -lxrt_coreutil -lpthread -lrt -ldl -lstdc++

all: xo
xo: $(BUILD_DIR)/slr_src.xo $(BUILD_DIR)/slr_sink.xo
$(BUILD_DIR)/slr_src.xo: slr_src.cpp hls_slr_stream.tcl
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) --temp_dir $(BUILD_DIR)/_x_compile_src --report_dir $(BUILD_DIR)/reports_compile_src --hls.pre_tcl $(abspath hls_slr_stream.tcl) -k slr_src -c -o $@ slr_src.cpp 2>&1 | tee $(BUILD_DIR)/slr_src_compile.log
$(BUILD_DIR)/slr_sink.xo: slr_sink.cpp hls_slr_stream.tcl
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) --temp_dir $(BUILD_DIR)/_x_compile_sink --report_dir $(BUILD_DIR)/reports_compile_sink --hls.pre_tcl $(abspath hls_slr_stream.tcl) -k slr_sink -c -o $@ slr_sink.cpp 2>&1 | tee $(BUILD_DIR)/slr_sink_compile.log
xclbin: $(BUILD_DIR)/slr_stream.xclbin
$(BUILD_DIR)/slr_stream.xclbin: $(BUILD_DIR)/slr_src.xo $(BUILD_DIR)/slr_sink.xo $(CLOCK_TCL)
	cd $(BUILD_DIR) && GEMV_TILE_SKIP_PBLOCK=1 GDN_KERNEL_TARGET_MHZ=$(FREQ) $(VPP) -l $(VPP_FLAGS) $(VPP_LDFLAGS) --temp_dir _x_temp --report_dir reports -o slr_stream.xclbin $(abspath $(BUILD_DIR)/slr_src.xo) $(abspath $(BUILD_DIR)/slr_sink.xo) 2>&1 | tee slr_stream_link.log
host: host_slr_stream.exe
host_slr_stream.exe: host_slr_stream.cpp
	$(CXX) -O2 -std=c++17 $(HOST_INC) -o $@ $< $(HOST_LIB)
clean:
	rm -rf $(BUILD_DIR) host_slr_stream.exe
.PHONY: all xo xclbin host clean

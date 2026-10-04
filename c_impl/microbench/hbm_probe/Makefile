# Iter78 step 0b: HBM read-bandwidth probe (32 readers, no compute).
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
OUTSTANDING   ?= 4
NPORTS        ?= 32
SLR           ?=
KERNEL        := hbm_probe
KSRC          := $(if $(filter 32,$(NPORTS)),hbm_probe.cpp,hbm_probe_np$(NPORTS).cpp)
PORTS         := $(shell seq 0 $$(( $(NPORTS) - 1 )))
BUILD_DIR     := build.$(TARGET).$(KERNEL).f$(FREQ).o$(OUTSTANDING)$(if $(filter 32,$(NPORTS)),,.np$(NPORTS))$(if $(SLR),.$(SLR),)
XO            := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN        := $(BUILD_DIR)/$(KERNEL).xclbin
HOST_BIN      := host_hbm_probe.exe
CLOCK_TCL     := $(abspath ../gemv_tile/apply_kernel_clock_target.tcl)

VPP_FLAGS     := -t $(TARGET) --platform $(PLATFORM) --save-temps --optimize 2
VPP_FLAGS     += --hls.jobs $(JOBS) --kernel_frequency $(FREQ)
VPP_CFLAGS    := $(VPP_FLAGS) -D PROBE_READ_OUTSTANDING=$(OUTSTANDING)
VPP_LDFLAGS   := --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
VPP_LDFLAGS   += --connectivity.nk $(KERNEL):1:$(KERNEL)_1
VPP_LDFLAGS   += $(foreach P,$(PORTS),--connectivity.sp $(KERNEL)_1.w$(P):HBM[$(P)])
VPP_LDFLAGS   += --connectivity.sp $(KERNEL)_1.out:HBM[0]
VPP_LDFLAGS   += $(if $(SLR),--connectivity.slr $(KERNEL)_1:$(SLR),)
# Vitis 2024.2 implements the kernel clock at 10 ns whatever --kernel_frequency
# says; the microbench's hook re-derives the MMCM ratio (GEMV_TILE_SKIP_PBLOCK=1
# keeps its floorplan out).
VPP_LDFLAGS   += --vivado.prop run.impl_1.STEPS.OPT_DESIGN.TCL.PRE=$(CLOCK_TCL)

HOST_INC      := -I$(XILINX_VITIS)/include -I$(XILINX_XRT)/include
HOST_LIB      := -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib -lxrt_coreutil -lpthread -lrt -ldl -lstdc++

all: xo

xo: $(XO)
$(XO): $(KSRC) hls_hbm_probe.tcl
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_CFLAGS) --temp_dir $(BUILD_DIR)/_x_compile --report_dir $(BUILD_DIR)/reports_compile \
	    --hls.pre_tcl $(abspath hls_hbm_probe.tcl) -k $(KERNEL) -c -o $@ $(KSRC) 2>&1 | tee $(BUILD_DIR)/$(KERNEL)_compile.log

xclbin: $(XCLBIN)
$(XCLBIN): $(XO) $(CLOCK_TCL)
	cd $(BUILD_DIR) && GEMV_TILE_SKIP_PBLOCK=1 GDN_KERNEL_TARGET_MHZ=$(FREQ) $(VPP) -l $(VPP_FLAGS) $(VPP_LDFLAGS) \
	    --temp_dir _x_temp --report_dir reports -o $(KERNEL).xclbin $(abspath $<) 2>&1 | tee $(KERNEL)_link.log

host: $(HOST_BIN)
$(HOST_BIN): host_hbm_probe.cpp
	$(CXX) -O2 -std=c++17 $(HOST_INC) -o $@ $< $(HOST_LIB)

clean:
	rm -rf $(BUILD_DIR) $(HOST_BIN)
.PHONY: all xo xclbin host clean

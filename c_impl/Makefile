# ============================================================================
# GatedDeltaNet c_impl/ — host testbench + v++ build for Alveo U55C.
#
# Builds three classes of artifact:
#   1. Host-side C++ testbenches that drive the HLS source directly through
#      its csim entry points (gdn_eval, gdn_attn_test, gdn_matmul_test).
#      These are what test_parity.sh runs.
#   2. The HLS kernel for U55C: gdn_forward.xo (compile) → gdn_forward.xclbin
#      (link). Targets are `xo` and `xclbin`; per-target output goes to
#      build.$(TARGET)/ so sw_emu/hw_emu/hw can coexist.
#   3. The XRT host program: host.exe — loads gdn_forward.xclbin onto the
#      board and runs a fixture through the hardware kernel.
#
# Modeled on cs356-project/matmul/{Makefile,pl/Makefile,host/Makefile},
# flattened into one file because c_impl/ keeps everything in one directory.
# ============================================================================

# ----------------------------------------------------------------------------
# Common host-toolchain config
# ----------------------------------------------------------------------------
CXX            ?= c++
# C++14 here matches the existing host testbench (gdn_eval etc.).
# -Wno-unknown-pragmas silences the Vitis HLS #pragma directives that the
# host gcc does not understand. -Wno-unused-label silences the HLS loop labels
# (pipeline/unroll targets) that are intentionally unused in the native build.
CXXFLAGS       ?= -O3 -std=c++14 -Wall -Wextra -pedantic -Wno-unknown-pragmas -Wno-unused-label

# Vitis HLS install location — provides hls_stream.h used by the systolic
# matmul kernel in gdn_model.cpp.
XILINX_HLS_INC ?= /tools/Xilinx/Vitis_HLS/2022.1/include


# ----------------------------------------------------------------------------
# Vitis / XRT install paths (override via env or `make XILINX_XRT=...`)
# ----------------------------------------------------------------------------
XILINX_VITIS   ?= /tools/Xilinx/Vitis/2022.1
XILINX_XRT     ?= /opt/xilinx/xrt
VPP            ?= $(XILINX_VITIS)/bin/v++


# ----------------------------------------------------------------------------
# v++ build knobs
# ----------------------------------------------------------------------------
# TARGET: sw_emu | hw_emu | hw
# FREQ:   kernel clock in MHz. Default 100 MHz to match the HLS source's
#         10 ns synthesis target. Without this, gen3x16 platforms (U55C) would
#         use the platform default of 300 MHz, leaving WNS ≈ -8.7 ns in the
#         router and making the design effectively unroutable.
PLATFORM       ?= xilinx_u55c_gen3x16_xdma_3_202210_1
TARGET         ?= hw
FREQ           ?= 100
JOBS           ?= 8

KERNEL         := gdn_forward
KERNEL_SRC     := gdn_model.cpp gdn_model.h
HW_CFG         := $(abspath ./hw.cfg)

BUILD_DIR      := build.$(TARGET)
XO             := $(BUILD_DIR)/$(KERNEL).xo
XCLBIN         := $(BUILD_DIR)/$(KERNEL).xclbin

XSA            := $(strip $(patsubst %.xpfm,%,$(shell basename $(PLATFORM))))

VPP_FLAGS      := -t $(TARGET) --platform $(PLATFORM) --save-temps
VPP_FLAGS      += -g --optimize 2
VPP_FLAGS      += --hls.jobs $(JOBS)
# --kernel_frequency overrides the platform's default kernel clock (300 MHz on
# U55C gen3x16) and matches the 100 MHz target the HLS source was synthesized
# for. Set unconditionally; v++ accepts it on both -c and -l. On non-gen3x16
# platforms it has the same effect as --clock.defaultFreqHz on the kernel
# clock domain.
VPP_FLAGS      += --kernel_frequency $(FREQ)

VPP_LDFLAGS    := --config $(HW_CFG)
VPP_LDFLAGS    += --vivado.synth.jobs $(JOBS) --vivado.impl.jobs $(JOBS)
# --profile.data inserts the DPA hardware trace IP (trace buffer on HBM[0]).
# At N=4 (8+ HBM masters spanning SLR0/1/2) that trace infrastructure crashes
# design-init: "HADAFileSet::getSrcOptions() : NULL pointer". We measure TPOT
# from host-side timing and correctness from the decode output, so the on-card
# profiling trace is not needed — leave it off. Re-enable (with
# --profile.trace_memory <bank>:<SLR>) only if port-bandwidth telemetry is wanted.
# VPP_LDFLAGS  += --profile.data all:all:all


# ----------------------------------------------------------------------------
# Host-program (XRT) build knobs
# ----------------------------------------------------------------------------
HOST_SRC       := host.cpp
HOST_BIN       := host.exe

HOST_INC       := -I$(XILINX_VITIS)/include -I$(XILINX_XRT)/include
# -Wl,-rpath bakes the XRT lib path into the binary so the runtime linker
# finds libxrt_*.so without needing `source /opt/xilinx/xrt/setup.sh`.
HOST_LIB       := -L$(XILINX_XRT)/lib -Wl,-rpath,$(XILINX_XRT)/lib \
                  -lxrt_core -lxrt_coreutil -lpthread -lrt -ldl -lstdc++


# ----------------------------------------------------------------------------
# On-card run knobs (consumed by the `run_hw` target)
# ----------------------------------------------------------------------------
#   WEIGHTS   flat .gdnw weight blob (~5.6 GB for GDN-1.3B)
#   FIXTURE   pretokenised .gdnreq fixture to score
#   HW_OUT    JSON result file (defaults to hw_<fixture-stem>.json)
#   HW_DEVICE XRT device index (0 if multiple U55Cs are on the host)
#   HW_MAX_EX optional cap on examples processed; leave empty for "all"
WEIGHTS        ?= artifacts/gdn-1.3b-f32.gdnw
FIXTURE        ?= fixtures_smoke/wikitext.gdnreq
HW_OUT         ?= hw_$(basename $(notdir $(FIXTURE))).json
HW_DEVICE      ?= 0
HW_MAX_EX      ?=


# ----------------------------------------------------------------------------
# Default target
# ----------------------------------------------------------------------------
.PHONY: all host_tb xo xclbin host run_hw clean clean_hw distclean help

all: host_tb

help:
	@echo "Targets:"
	@echo "  host_tb     C++ host testbenches: gdn_eval gdn_attn_test gdn_matmul_test (default)"
	@echo "  xo          v++ compile  → $(XO)"
	@echo "  xclbin      v++ link     → $(XCLBIN)"
	@echo "  host        XRT host    → $(HOST_BIN)"
	@echo "  run_hw      xo + xclbin + host + ./host.exe ... (full hardware run)"
	@echo "  clean       remove host-tb binaries"
	@echo "  clean_hw    remove build.\$$(TARGET), _x/, v++ logs"
	@echo "  distclean   clean + clean_hw + remove host.exe"
	@echo ""
	@echo "Build knobs (override on cmdline, e.g. make TARGET=hw_emu):"
	@echo "  PLATFORM=$(PLATFORM)"
	@echo "  TARGET=$(TARGET)        (sw_emu | hw_emu | hw)"
	@echo "  FREQ=$(FREQ) MHz"
	@echo "  JOBS=$(JOBS)"
	@echo ""
	@echo "run_hw knobs (override on cmdline, e.g. make run_hw FIXTURE=fixtures_smoke/piqa.gdnreq):"
	@echo "  WEIGHTS=$(WEIGHTS)"
	@echo "  FIXTURE=$(FIXTURE)"
	@echo "  HW_OUT=$(HW_OUT)"
	@echo "  HW_DEVICE=$(HW_DEVICE)"
	@echo "  HW_MAX_EX=$(if $(HW_MAX_EX),$(HW_MAX_EX),(all))"


# ----------------------------------------------------------------------------
# C++ host testbenches
# ----------------------------------------------------------------------------
# Decode-only build: gdn_eval is the native csim of the decode-only GEMV kernel.
# The old gdn_attn_test / gdn_matmul_test exercised the removed prefill /
# GEMM-matmul tops (gdn_attn_forward, gdn_matmul_top) and are retired.
host_tb: gdn_eval

gdn_eval: gdn_eval.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_eval.cpp gdn_model.cpp -lm

gdn_attn_test: gdn_attn_test.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ gdn_attn_test.cpp gdn_model.cpp -lm

# gdn_matmul_test exercises the systolic matmul through the gdn_matmul_top
# HLS entry in gdn_model.cpp, comparing against the native-C naive_matmul
# reference in gdn_matmul_test.cpp. Same translation unit as the rest of
# the model — no separate matmul source file.
gdn_matmul_test: gdn_matmul_test.cpp gdn_model.cpp gdn_model.h
	$(CXX) $(CXXFLAGS) -I$(XILINX_HLS_INC) -o $@ \
	    gdn_matmul_test.cpp gdn_model.cpp -lm


# ----------------------------------------------------------------------------
# v++ compile: gdn_model.cpp → $(KERNEL).xo
# ----------------------------------------------------------------------------
xo: $(XO)

$(XO): $(KERNEL_SRC)
	@mkdir -p $(BUILD_DIR)
	$(VPP) $(VPP_FLAGS) -k $(KERNEL) -c -o $@ $(filter %.cpp,$(KERNEL_SRC)) \
	    2>&1 | tee $(BUILD_DIR)/$(KERNEL)_compile.log


# ----------------------------------------------------------------------------
# v++ link: $(KERNEL).xo + hw.cfg → $(KERNEL).xclbin
# ----------------------------------------------------------------------------
xclbin: $(XCLBIN)

$(XCLBIN): $(XO) $(HW_CFG)
	@mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && $(VPP) -l $(VPP_FLAGS) $(VPP_LDFLAGS) \
	    --temp_dir _x_temp/ \
	    --report_dir reports/ \
	    -o $(KERNEL).xclbin \
	    $(abspath $<) \
	    2>&1 | tee $(KERNEL)_link.log


# ----------------------------------------------------------------------------
# XRT host program
# ----------------------------------------------------------------------------
host: $(HOST_BIN)

# Links gdn_model.cpp for the host-side weight-shard builder
# (gdn_build_weight_shards / gdn_weight_shard_floats). gdn_model.cpp needs the
# Vitis HLS include for hls_stream.h; -lm for the libm calls it pulls in.
$(HOST_BIN): $(HOST_SRC) gdn_model.cpp gdn_model.h
	$(CXX) -O2 -std=c++17 -Wall -Wno-unknown-pragmas -Wno-unused-label \
	    $(HOST_INC) -I$(XILINX_HLS_INC) -o $@ $(HOST_SRC) gdn_model.cpp $(HOST_LIB) -lm


# ----------------------------------------------------------------------------
# End-to-end hardware run
# ----------------------------------------------------------------------------
# Builds the .xo (if missing), links the .xclbin (if missing), builds the
# host program (if missing), then invokes host.exe to score $(FIXTURE) on the
# Alveo card. Reuses cached build artefacts when the prerequisite files have
# not changed, so a re-run after only editing $(FIXTURE) skips straight to
# the on-card invocation.
run_hw: $(XCLBIN) $(HOST_BIN)
	@echo "=== Running on hardware ==="
	@echo "  xclbin : $(XCLBIN)"
	@echo "  weights: $(WEIGHTS)"
	@echo "  fixture: $(FIXTURE)"
	@echo "  output : $(HW_OUT)"
	@echo "  device : $(HW_DEVICE)"
	@echo "  max_ex : $(if $(HW_MAX_EX),$(HW_MAX_EX),(all))"
	./$(HOST_BIN) $(XCLBIN) $(WEIGHTS) $(FIXTURE) $(HW_OUT) $(HW_DEVICE) $(HW_MAX_EX)


# ----------------------------------------------------------------------------
# Clean
# ----------------------------------------------------------------------------
clean:
	rm -f gdn_eval gdn_attn_test gdn_matmul_test

clean_hw:
	rm -rf $(BUILD_DIR) _x v++_*.log v++_*.jou .Xil

distclean: clean clean_hw
	rm -f $(HOST_BIN)

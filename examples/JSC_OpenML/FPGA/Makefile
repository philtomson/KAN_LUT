# examples/JSC_OpenML/FPGA/Makefile
# Makefile for compiling and running the JSC_OpenML Generic KAN IP Core simulation

JULIA = julia
IVERILOG = iverilog

CORE_DIR = ../../../hardware
CORE_SRC = $(CORE_DIR)/kan_generic_core.sv
BRAM_SRC = $(CORE_DIR)/kan_bram_bank.sv

TB_SRC = tb_jsc_core.sv
TOP_SRC = jsc_generic_top.sv

SIM_BIN = sim_jsc_core
MEM_FILES = layer1_lane0.mem layer1_lane1.mem layer1_lane2.mem layer1_lane3.mem \
            layer2_lane0.mem layer2_lane1.mem layer2_lane2.mem layer2_lane3.mem

.PHONY: all run compile generate clean

all: run

run: compile
	@echo "=== Running JSC_OpenML Generic KAN Core Simulation ==="
	./$(SIM_BIN) +DATA_FILE=tb_data.txt

compile: $(SIM_BIN)

$(SIM_BIN): $(TB_SRC) $(TOP_SRC) $(CORE_SRC) $(BRAM_SRC) $(MEM_FILES) tb_data.txt
	@echo "=== Compiling Generic Core with iverilog ==="
	$(IVERILOG) -g2012 -o $(SIM_BIN) $(TB_SRC) $(TOP_SRC) $(CORE_SRC) $(BRAM_SRC)

generate:
	@echo "=== Generating BRAM Lane Memory Files ==="
	$(JULIA) --project=../../.. generate_mem_files.jl
	@echo "=== Generating Simulation Test Vectors ==="
	$(JULIA) --project=../../.. generate_test_vectors.jl

tb_data.txt $(MEM_FILES):
	$(MAKE) generate

clean:
	@echo "=== Cleaning Simulation Artifacts ==="
	rm -f $(SIM_BIN) $(MEM_FILES) tb_data.txt

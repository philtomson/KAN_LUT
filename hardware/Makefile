# hardware/Makefile
# Makefile for compiling and running the Parameterized Generic KAN IP Core simulation

JULIA = julia
IVERILOG = iverilog

TB_SRC = tb_generic_core.sv
TOP_SRC = mnist_generic_top.sv
CORE_SRC = kan_generic_core.sv
BRAM_SRC = kan_bram_bank.sv

SIM_BIN = sim_generic_core
MEM_FILES = layer1_lane0.mem layer1_lane1.mem layer1_lane2.mem layer1_lane3.mem \
            layer2_lane0.mem layer2_lane1.mem layer2_lane2.mem layer2_lane3.mem

.PHONY: all run compile generate clean help

all: run

run: compile
	@echo "=== Running Generic KAN Core Simulation ==="
	./$(SIM_BIN) +DATA_FILE=../examples/MNIST/FPGA/tb_data.txt

compile: $(SIM_BIN)

$(SIM_BIN): $(TB_SRC) $(TOP_SRC) $(CORE_SRC) $(BRAM_SRC) $(MEM_FILES)
	@echo "=== Compiling Generic Core with iverilog ==="
	$(IVERILOG) -g2012 -o $(SIM_BIN) $(TB_SRC) $(TOP_SRC) $(CORE_SRC) $(BRAM_SRC)

generate:
	@echo "=== Generating BRAM Lane Memory Files ==="
	$(JULIA) --project=.. generate_mem_files.jl

clean:
	@echo "=== Cleaning Simulation Artifacts ==="
	rm -f $(SIM_BIN)

help:
	@echo "Generic KAN Core Simulation Makefile"
	@echo "Targets:"
	@echo "  all       : Compile and run simulation (default)"
	@echo "  compile   : Compile SystemVerilog files into $(SIM_BIN)"
	@echo "  run       : Execute compiled simulation binary"
	@echo "  generate  : Run Julia script to regenerate BRAM mem initialization files"
	@echo "  clean     : Remove generated simulation binaries"

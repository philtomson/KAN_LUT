# examples/MNIST/FPGA/Makefile
# Makefile for compiling and running MNIST KAN-LUT FPGA Simulation

JULIA = julia
IVERILOG = iverilog
VVP = vvp

# Sources
TB_SRC = tb_mnist_kan.sv
TOP_SRC = mnist_kan_top.sv
LAYER1_SRC = mnist_kan_layer1.sv
LAYER2_SRC = mnist_kan_layer2.sv

SIM_BIN = sim_mnist_kan

.PHONY: all run compile generate clean synth help

all: run

run: compile
	@echo "=== Running Pipelined KAN-LUT Simulation ==="
	./$(SIM_BIN) +DATA_FILE=tb_data.txt

compile: $(SIM_BIN)

$(SIM_BIN): $(TB_SRC) $(TOP_SRC) $(LAYER1_SRC) $(LAYER2_SRC)
	@echo "=== Compiling SystemVerilog Design with iverilog ==="
	$(IVERILOG) -g2012 -o $(SIM_BIN) $(TB_SRC) $(TOP_SRC) $(LAYER1_SRC) $(LAYER2_SRC)

generate:
	@echo "=== Regenerating RTL from Trained LUTs ==="
	$(JULIA) --project=../.. generate_rtl.jl
	@echo "=== Regenerating Test Vectors from Dataset ==="
	$(JULIA) --project=../.. generate_test_vectors.jl

clean:
	@echo "=== Cleaning Simulation Artifacts ==="
	rm -f $(SIM_BIN)

synth:
	@echo "=== Synthesizing Custom MNIST RTL with Yosys ==="
	yosys -p "read_verilog -sv $(TOP_SRC) $(LAYER1_SRC) $(LAYER2_SRC); synth_ecp5 -top mnist_kan_top"

help:
	@echo "MNIST KAN-LUT FPGA Simulation Makefile"
	@echo "Targets:"
	@echo "  all       : Compile and run simulation (default)"
	@echo "  compile   : Compile SystemVerilog files into $(SIM_BIN)"
	@echo "  run       : Execute compiled simulation binary"
	@echo "  generate  : Run Julia scripts to regenerate RTL and test vectors"
	@echo "  clean     : Remove generated simulation binaries"
	@echo "  synth     : Synthesize the design using Yosys for ECP5"

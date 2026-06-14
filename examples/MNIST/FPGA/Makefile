# examples/MNIST/FPGA/Makefile
# Makefile for compiling and running MNIST KAN-LUT FPGA Simulation using Verilator

JULIA = julia
VERILATOR = verilator

# Sources for the unrolled, mixed-precision KAN core
TOP_SRC = mnist_kan_top_mixed.sv
LAYER1_SRC = mnist_kan_layer1_mixed.sv
LAYER2_SRC = mnist_kan_layer2_mixed.sv

TB_CPP = tb_mnist_kan_mixed.cpp
SIM_INTERACTIVE_CPP = sim_interactive_mixed.cpp

SDL2_CFLAGS := $(shell pkg-config --cflags sdl2)
SDL2_LIBS   := $(shell pkg-config --libs   sdl2)

.PHONY: all run compile generate clean sim_interactive run_8bit compile_8bit help

all: run

run: compile
	@echo "=== Running Verilator Pipelined KAN-LUT Simulation ==="
	./obj_dir/Vmnist_kan_top +DATA_FILE=tb_data_mixed.txt

compile: obj_dir/Vmnist_kan_top

obj_dir/Vmnist_kan_top: $(TB_CPP) $(TOP_SRC) $(LAYER1_SRC) $(LAYER2_SRC)
	@echo "=== Compiling SystemVerilog Design with Verilator ==="
	$(VERILATOR) --cc --exe --build -Wno-fatal \
		$(TB_CPP) $(TOP_SRC) $(LAYER1_SRC) $(LAYER2_SRC) \
		--top mnist_kan_top -o Vmnist_kan_top

# 8-bit fully unrolled simulation targets
run_8bit: compile_8bit
	@echo "=== Running Verilator Pipelined 8-bit KAN Simulation ==="
	./obj_dir_8bit/Vtb_mnist_kan +DATA_FILE=tb_data.txt

compile_8bit: obj_dir_8bit/Vtb_mnist_kan

obj_dir_8bit/Vtb_mnist_kan: tb_mnist_kan.sv mnist_kan_top.sv mnist_kan_layer1.sv mnist_kan_layer2.sv
	@echo "=== Compiling 8-bit SystemVerilog Design with Verilator ==="
	$(VERILATOR) --binary -Wno-fatal \
		tb_mnist_kan.sv mnist_kan_top.sv mnist_kan_layer1.sv mnist_kan_layer2.sv \
		--top tb_mnist_kan -o Vtb_mnist_kan --Mdir obj_dir_8bit

generate:
	@echo "=== Regenerating RTL from Trained LUTs ==="
	$(JULIA) --project=../../.. generate_mixed_rtl.jl
	$(JULIA) --project=../../.. generate_rtl.jl
	@echo "=== Regenerating Test Vectors from Dataset ==="
	$(JULIA) --project=../../.. generate_mixed_test_vectors.jl
	$(JULIA) --project=../../.. generate_test_vectors.jl

sim_interactive: obj_dir_interactive/Vmnist_kan_top_interactive
	@echo "=== Launching interactive simulation ==="
	./obj_dir_interactive/Vmnist_kan_top_interactive

obj_dir_interactive/Vmnist_kan_top_interactive: $(SIM_INTERACTIVE_CPP) $(TOP_SRC) $(LAYER1_SRC) $(LAYER2_SRC)
	@echo "=== Building interactive simulation (mnist_kan_top + SDL2) ==="
	rm -rf obj_dir_interactive
	$(VERILATOR) --cc --exe --build -Wno-fatal \
		--CFLAGS "$(SDL2_CFLAGS)" --LDFLAGS "$(SDL2_LIBS)" \
		$(SIM_INTERACTIVE_CPP) $(TOP_SRC) $(LAYER1_SRC) $(LAYER2_SRC) \
		--top mnist_kan_top -o Vmnist_kan_top_interactive --Mdir obj_dir_interactive

nano20k_mixed:
	@echo "=== Synthesizing and Place/Routing for Tang Nano 20K (Mixed-Precision Unrolled) ==="
	yosys -p "read_verilog -sv nano20k_top_mixed.sv mnist_kan_top_mixed.sv mnist_kan_layer1_mixed.sv mnist_kan_layer2_mixed.sv ../../../hardware/uart_rx.v ../../../hardware/uart_tx.v ../../../hardware/gowin_rpll/gowin_rpll.v; synth_gowin -top nano20k_top_mixed -json nano20k_mixed.json"
	nextpnr-himbaechel --json nano20k_mixed.json --write nano20k_mixed_pnr.json --device GW2AR-LV18QN88C8/I7 --vopt family=GW2A-18C --vopt cst=nano20k.cst
	gowin_pack -d GW2A-18C -o pack_mixed.fs nano20k_mixed_pnr.json

clean:
	@echo "=== Cleaning Simulation Artifacts ==="
	rm -rf obj_dir obj_dir_interactive obj_dir_8bit
	rm -f nano20k_mixed.json nano20k_mixed_pnr.json pack_mixed.fs

help:
	@echo "MNIST KAN-LUT FPGA Simulation Makefile (Verilator & Gowin)"
	@echo "Targets:"
	@echo "  all             : Compile and run mixed-precision simulation (default)"
	@echo "  compile         : Compile mixed SystemVerilog files into Verilator executable"
	@echo "  run             : Execute compiled mixed-precision simulation binary"
	@echo "  run_8bit        : Compile and run 8-bit unrolled simulation binary"
	@echo "  generate        : Run Julia scripts to regenerate all RTL and test vectors"
	@echo "  sim_interactive : Compile and launch interactive SDL2/Verilator drawing application"
	@echo "  nano20k_mixed   : Synthesize and place-and-route unrolled mixed KAN for Tang Nano 20K"
	@echo "  clean           : Remove generated verilator and physical synthesis outputs"

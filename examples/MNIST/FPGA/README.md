# MNIST KAN-LUT FPGA Deployments

This directory contains application-specific FPGA implementations of the trained MNIST KAN model. Two different hardware paradigms are supported:

1.  **Paradigm 1: Fully-Unrolled Hardcoded RTL** (Low latency, high area/BRAM)
2.  **Paradigm 2: Parameterized Generic IP Core** (Memory-mapped weights from SDRAM, low area, configurable)

---

## Directory Overview

*   **Paradigm 1 (Unrolled)**:
    *   `mnist_kan_top.sv` / `mnist_kan_layer1.sv` / `mnist_kan_layer2.sv`: Generated Verilog modules containing unrolled combinational LUT lookup arrays.
    *   `tb_mnist_kan.sv`: Simulation testbench for Paradigm 1.
    *   `Makefile`: Build automation for compiling and simulating Paradigm 1.
*   **Paradigm 2 (Generic & Tang Nano 20K)**:
    *   `mnist_generic_top.sv`: Chains two generic `kan_generic_core` layers with memory offsets.
    *   `nano20k_top.sv`: Top-level Sipeed Tang Nano 20K (Gowin GW2AR-18C) physical integration wrapping the generic core, implementing SDRAM interface and 2 Mbaud UART loader.
    *   `nano20k.cst`: Pin constraints file for Tang Nano 20K.
    *   `tb_generic_core.sv`: Standard testbench verifying the generic core with mock PSRAM.
    *   `sim_interactive.cpp`: Verilator + SDL2 interactive drawing canvas simulation.
    *   `test_nano20k.py`: Python script to pack trained weights and stream them over UART for hardware verification.
    *   `generate_mem_files.jl`: Julia script to convert LUT JSON files to hex `.mem` files for simulation initialization.
    *   `Makefile.generic`: Build automation for compiling, simulating, and synthesizing the generic design targeting the Tang Nano 20K.

---

## 1. Paradigm 2: Simulation & Verification

### Standard Testbench Simulation
To compile and execute the testbench using Iverilog:
```bash
make -f Makefile.generic generate  # Generate lane memory hex files
make -f Makefile.generic compile   # Compile tb_generic_core.sv
make -f Makefile.generic run       # Run simulation against tb_data.txt
```

### Interactive Drawing Simulator (SDL2 + Verilator)
To build and launch the GUI window where you can draw digits with your mouse:
```bash
make -f Makefile.generic sim_interactive
```

---

## 2. Paradigm 2: Tang Nano 20K Deployment

### Synthesize & Place-and-Route
Ensure you have the **OSS CAD Suite** (Yosys, nextpnr-himbaechel, and apycula/gowin_pack) installed. Then run:
```bash
make -f Makefile.generic nano20k
```
This produces the final bitstream file `pack.fs`.

### Flash to Board
```bash
openFPGALoader -b tangnano20k pack.fs
```

### Stream Weights & Test Inference
Run the automated host script to stream the trained JSON weights (6.75 MB) and evaluate prediction accuracy:
```bash
python test_nano20k.py /dev/ttyUSB1
```
*(Note: Replace `/dev/ttyUSB1` with the correct serial port of your board).*

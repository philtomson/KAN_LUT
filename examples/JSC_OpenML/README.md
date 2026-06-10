# JSC_OpenML Jet Substructure Classification

This directory contains the **Jet Substructure Classification (JSC)** example, ported from the `KANELE` repository, implemented using the generic parameterizable KAN IP cores from `hardware/`.

This example demonstrates the parameterizability of the generic KAN IP core:
*   **Input Features**: 16 jet shape observables.
*   **Output Classes**: 5 categories (quark, gluon, W, Z, top).
*   **Network Architecture**: 2-layer KAN `[16 -> 8 -> 5]`.
*   **Bit-Accuracy**: Verified to achieve 100% bit-accurate predictions between the fixed-point Julia inference model and the SystemVerilog hardware simulation.

---

## Directory Structure

*   `data/`: Contains symlinks to the dataset `.npy` files.
*   `demo.jl`: Julia script to load the data, train the `[16 -> 8 -> 5]` KAN model, discretize it to 8-bit LUTs, and export them as JSON.
*   `FPGA/jsc_generic_top.sv`: Top-level SystemVerilog wrapper mapping the generic KAN cores to a contiguous weight memory layout.
*   `FPGA/tb_jsc_core.sv`: SystemVerilog testbench utilizing a cycle-accurate mock memory interface.
*   `FPGA/generate_mem_files.jl`: Partitioning script to split JSON LUT values into lane-specific `.mem` files.
*   `FPGA/generate_test_vectors.jl`: Generates test stimulus (`tb_data.txt`) matching the integer model predictions.
*   `FPGA/Makefile`: Verification build script.

---

## How to Run

Ensure you are in the project root directory:
```bash
cd /home/phil/devel/FPGA/KAN_LUT
```

### 1. Train and Discretize the KAN
Run the training script (takes ~1-2 minutes):
```bash
julia --project=. examples/JSC_OpenML/demo.jl
```
This trains the KAN model on the OpenML dataset and outputs:
*   `jsc_luts_layer1.json`
*   `jsc_luts_layer2.json`

### 2. Run FPGA Verification (Icarus Verilog)
Change to the FPGA subdirectory and run the Makefile:
```bash
cd examples/JSC_OpenML/FPGA
make clean && make
```
This target will:
1.  Partition the trained JSON LUT files into lane-specific `.mem` files.
2.  Generate 100 test stimulus vectors.
3.  Compile the SystemVerilog source files with Icarus Verilog.
4.  Run the simulation, verifying that all 100 hardware predictions exactly match the software model.

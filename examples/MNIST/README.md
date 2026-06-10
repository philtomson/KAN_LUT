# MNIST Digits Classification Demo

This directory implements KAN training, 8-bit L-LUT discretization, and bit-accurate inference on the MNIST handwritten digit classification task.

To make the KAN extremely lightweight and fast to train, 28x28 pixel images are downsampled to 14x14 pixel images ($196$ input features) using $2 \times 2$ average pooling.

---

## Files

1. **`demo.jl`**:
   * Downloads and loads the MNIST dataset using `MLDatasets.jl`.
   * Downsamples the dataset to 14x14 pixels.
   * Trains a 2-layer KAN (`196 -> 32 -> 10`) on GPU (`AMDGPU.jl` or CPU).
   * Discretizes the trained KAN to static 8-bit Look-Up Tables (L-LUTs).
   * Saves the tables to `mnist_luts_layer1.json` and `mnist_luts_layer2.json`.
   * Validates and prints the final continuous model test accuracy vs. the bit-accurate integer test accuracy.
2. **`inference.jl`**:
   * Loads the saved JSON LUTs from `demo.jl`.
   * Fetches specific digits from the MNIST test set.
   * Prints the digits directly in the terminal using ASCII art.
   * Performs the exact integer LUT inference matching FPGA hardware behavior and outputs the classified digit.
3. **`FPGA/`**:
   * Contains the **Paradigm 1 (Fully-Unrolled Custom RTL)** implementation and simulation environment.

---

## FPGA Implementation Paradigms

We provide two distinct hardware implementations of the MNIST KAN accelerator:
1. **Paradigm 1 (Fully-Unrolled Custom RTL)**: Located in [examples/MNIST/FPGA/](file:///home/phil/devel/FPGA/KAN_LUT/examples/MNIST/FPGA). A Julia script automatically generates a structural, pipelined, single-cycle throughput design tailored to the specific MNIST weights.
2. **Paradigm 2 (Parameterized Generic IP Core)**: Also located in [examples/MNIST/FPGA/](file:///home/phil/devel/FPGA/KAN_LUT/examples/MNIST/FPGA). A multi-cycle time-multiplexed accelerator that reads weight layers sequentially from memory (such as SDRAM or PSRAM) and is fully runtime-parameterized.

### Performance & Resource Comparison

| Metric | Paradigm 1: Fully-Unrolled RTL | Paradigm 2: Parameterized Generic IP Core (P=4) |
| :--- | :---: | :---: |
| **Datapath Style** | Fully Parallel & Structural | Time-Multiplexed Accumulating |
| **Inference Latency** | **15 clock cycles** | **1,734 clock cycles** |
| **Throughput** | **1 sample / clock cycle** | **1 sample / 1,734 clock cycles** |
| **Logic Footprint (LUTs/LEs)** | Massive (Est. 50,000+ LEs) | Extremely Small (Est. < 1,000 LEs) |
| **Physical Adder Count** | **6,550 adders** | **8 adders** (two 4-lane trees + accumulators) |
| **BRAM Block Usage** | None (ROMs unrolled into logic) | **~23.5 Megabits** of BRAM storage |
| **Weights Reloading** | Requires re-bitstream compilation | Dynamic bootloading via write ports |
| **Design Flexibility** | Hardcoded to network dimensions | Fully run-time parameterized |

### The Resource Trade-off: Logic Pruning vs. BRAM Allocation
The raw weight data for the `196 -> 64 -> 10` network is exactly **6.75 Megabytes**:
* **Layer 1**: $196 \text{ inputs} \times 64 \text{ outputs} \times 256 \text{ entries} \times 14 \text{ bits/entry} \approx 44.96\text{ Mbits}$.
* **Layer 2**: $64 \text{ inputs} \times 10 \text{ outputs} \times 256 \text{ entries} \times 13 \text{ bits/entry} \approx 2.12\text{ Mbits}$.

* In **Paradigm 1**, weights are hardcoded as constant ROM lookups. The compiler (e.g., Yosys or Vivado) applies **Boolean minimization** and **logic pruning**, removing zero-weights and matching duplicate paths. This dramatically reduces the actual logic footprint.
* In **Paradigm 2**, the memory banks are writeable, allowing model reloading at boot time. Because weights can change, the compiler **cannot** prune the storage. It must allocate the full physical BRAM capacity, which makes the design light on logic gates but memory-intensive.

### Classification Accuracy

Both Verilog implementations achieve the **exact same classification accuracy** of **93.66%** (on the full MNIST test set), as they implement identical quantized mathematical logic:
* **Continuous Float KAN Model**: **95.62%** accuracy.
* **Bit-Accurate Verilog (Paradigm 1 & 2)**: **93.66%** accuracy.
* **Quantization Drop**: Only **1.96%** accuracy drop going from floating-point training to 8-bit integer hardware LUT inference.
* **RTL Verification Parity**: The testbenches verify both designs against 100 test samples and confirm **100% bit-accurate predictions** (zero mismatches) relative to the reference software integer model.

---

## How to Run

Ensure you are in the project root directory:
```bash
cd /home/phil/devel/FPGA/KAN_LUT
```

### 1. Train and Discretize the KAN
Run the training script (takes ~1-2 minutes to download, compile, train, and export):
```bash
julia --project=. examples/MNIST/demo.jl
```

### 2. Run Custom Inference on Test Digits
After training has generated the JSON files, run the inference script to visualize digits in ASCII and classify them:
```bash
julia --project=. examples/MNIST/inference.jl
```

### 3. Run FPGA Simulations (Icarus Verilog)

First, generate the RTL files (for Paradigm 1) and memory files (for Paradigm 2) from the trained JSON:
```bash
# Generate Paradigm 1 RTL
julia --project=. examples/MNIST/FPGA/generate_rtl.jl

# Generate Paradigm 2 Memory Files
julia --project=. examples/MNIST/FPGA/generate_mem_files.jl

# Generate Test Stimulus Vector (100 MNIST samples)
julia --project=. examples/MNIST/FPGA/generate_test_vectors.jl
```

#### Run Paradigm 1 (Fully-Unrolled RTL) Simulation:
```bash
cd examples/MNIST/FPGA
make clean && make run
```
* **Expected Output**: `SUCCESS: All 100 samples verified with 100% bit-accurate parity!`

#### Run Paradigm 2 (Generic IP Core) Simulation:
```bash
cd examples/MNIST/FPGA
make -f Makefile.generic clean && make -f Makefile.generic run
```
* **Expected Output**: `SUCCESS: All 100 samples verified with 100% bit-accurate parity on PSRAM-backed KAN Core!`

---

## 4. Tang Nano 20K Physical FPGA Deployment

Paradigm 2 supports physical synthesis and deployment on the **Sipeed Tang Nano 20K** using its embedded 64Mbit SDRAM.

1.  **Synthesize and Place-and-Route**:
    ```bash
    cd examples/MNIST/FPGA
    make -f Makefile.generic nano20k
    ```
    This compiles the design using Yosys and nextpnr, producing the bitstream `pack.fs`.

2.  **Flash the FPGA**:
    ```bash
    openFPGALoader -b tangnano20k pack.fs
    ```

3. **Stream Weights & Verify Inference**:
    ```bash
    python test_nano20k.py /dev/ttyUSB1
    ```
    This script converts the trained JSON LUT weights to a flat 6.75 MB binary payload, uploads it to the SDRAM over a 2 Mbaud UART connection, and streams test images to verify physical inference accuracy.

---

## 5. Interactive Verilator Simulation (SDL2 Canvas)

You can launch an interactive desktop application where you draw digits with your mouse on an SDL2 window, and the Verilator simulation of our generic KAN IP Core runs inference on it in real time.

### Installing Dependencies

Ensure you have Verilator, SDL2, and compiler tools installed:

#### On Ubuntu / Debian:
```bash
sudo apt-get update
sudo apt-get install verilator libsdl2-dev pkg-config build-essential
```

#### On Fedora:
```bash
sudo dnf install verilator SDL2-devel pkgconf-pkg-config make gcc-c++
```

### Running the Interactive Simulation

To build and launch the SDL2-backed Verilator simulation:
```bash
cd examples/MNIST/FPGA
make -f Makefile.generic sim_interactive
```
This will:
1. Compile the SystemVerilog core using Verilator.
2. Link it against the host C++ SDL2 drawing canvas (`sim_interactive.cpp`).
3. Open a window for drawing digits. The simulated KAN will classify your drawings live!

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
2. **`qat_mixed_demo.jl`**:
   * Trains a mixed-precision QAT KAN (`[1, 6, 6]` bits) on 14x14 downsampled MNIST features.
   * Runs an asymptotic pruning schedule that ramps up to a threshold of `0.15` by Epoch 15.
   * Discretizes the trained KAN to variable-sized L-LUTs and validates them via bit-accurate integer inference.
3. **`inference.jl`**:
   * Loads the saved JSON LUTs from `demo.jl`.
   * Fetches specific digits from the MNIST test set.
   * Prints the digits directly in the terminal using ASCII art.
   * Performs the exact integer LUT inference matching FPGA hardware behavior and outputs the classified digit.
3. **`FPGA/`**:
   * Contains the **Paradigm 1 (Fully-Unrolled Custom RTL)** implementation and simulation environment.

---

## FPGA Implementation Paradigms

We provide two distinct hardware implementations of the MNIST KAN accelerator:
1. **Paradigm 1 (Fully-Unrolled Custom RTL)**: Located in [examples/MNIST/FPGA/](file:///home/phil/devel/FPGA/KAN_LUT/examples/MNIST/FPGA). A Julia script automatically generates a structural, pipelined, single-cycle throughput design tailored to the specific MNIST weights. Supported in both standard **8-bit** and optimized **mixed-precision [1, 6, 6] bits** mode.
2. **Paradigm 2 (Parameterized Generic IP Core)**: Also located in [examples/MNIST/FPGA/](file:///home/phil/devel/FPGA/KAN_LUT/examples/MNIST/FPGA). A multi-cycle time-multiplexed accelerator that reads weight layers sequentially from memory (such as SDRAM or PSRAM) and is fully runtime-parameterized.

### Performance & Resource Comparison

| Metric | Paradigm 1: Fully-Unrolled (8-bit) | Paradigm 1: Fully-Unrolled (Mixed [1,6,6]) | Paradigm 2: Parameterized Generic Core (P=4) |
| :--- | :---: | :---: | :---: |
| **Datapath Style** | Fully Parallel & Structural | Fully Parallel & Structural (Combinational Adders) | Time-Multiplexed Accumulating |
| **Inference Latency** | **16 clock cycles** | **2 clock cycles** | **1,734 clock cycles** |
| **Throughput** | **1 sample / clock cycle** | **1 sample / clock cycle** | **1 sample / 1,734 clock cycles** |
| **Logic Footprint (LUTs)** | Massive (Est. 50,000+ LUT4) | **7,000 - 9,000 LUT4** | **8,785 LUT4** (incl. SDRAM/UART/FSM wrapper) |
| **DFF Register Count** | Est. 20,000+ DFFs | **438 DFFs** | **4,472 DFFs** (incl. SDRAM/UART/FSM wrapper) |
| **Physical Adder Count** | **6,550 adders** | **1,650 adders** (combinational) | **8 adders** (two 4-lane trees + accumulators) |
| **BRAM Block Usage** | None (ROMs unrolled to logic) | None (ROMs unrolled to logic) | **0 BSRAMs** (weights stored in external SDRAM) |
| **Weights Reloading** | Requires re-bitstream compilation | Requires re-bitstream compilation | Dynamic bootloading over UART to SDRAM |
| **Design Flexibility** | Hardcoded to network weights | Hardcoded to network weights | Fully run-time parameterized |
| **Tang Nano 20K Fit** | **No** (exceeds LUT capacity) | **Yes** (fits comfortably at ~40% LUT / 3% DFF) | **Yes** (fits comfortably at 42% LUT / 28% DFF) |

### The Resource Trade-off: Logic Pruning vs. BRAM Allocation
The raw weight data for the `196 -> 64 -> 10` network is exactly **6.75 Megabytes**:
* **Layer 1**: $196 \text{ inputs} \times 64 \text{ outputs} \times 256 \text{ entries} \times 14 \text{ bits/entry} \approx 44.96\text{ Mbits}$.
* **Layer 2**: $64 \text{ inputs} \times 10 \text{ outputs} \times 256 \text{ entries} \times 13 \text{ bits/entry} \approx 2.12\text{ Mbits}$.

* In **Paradigm 1**, weights are hardcoded as constant ROM lookups. The compiler (e.g., Yosys or Vivado) applies **Boolean minimization** and **logic pruning**, removing zero-weights and matching duplicate paths. This dramatically reduces the actual logic footprint.
  * In the **8-bit** configuration, the lookups have 256 entries. This creates massive tables (63.6 MB Verilog file size) which are expensive to compile (~10 mins) and synthesize.
  * In the **mixed-precision [1, 6, 6]** configuration, Layer 1 ROM lookup tables require only 2 entries (since inputs are 1-bit binary). By implementing **combinational adder trees** (no pipeline registers inside the tree), the register footprint is reduced to just the layer output registers (**438 DFFs**). This allows the entire design to fit comfortably within the Sipeed Tang Nano 20K!
* In **Paradigm 2**, the memory banks are writeable, allowing model reloading at boot time. Because weights can change, the compiler **cannot** prune the storage. In standard BRAM configuration, this requires massive block RAM resource. However, our physical implementation bypasses this by streaming weights dynamically from the board's onboard SDRAM/PSRAM, utilizing **0 BRAM blocks** for weight storage.

### Classification Accuracy

Both Verilog implementations achieve the **exact same classification accuracy** of **93.66%** (on the full MNIST test set), as they implement identical quantized mathematical logic:
* **Continuous Float KAN Model**: **95.62%** accuracy.
* **Bit-Accurate Verilog (Paradigm 1 & 2)**: **93.66%** accuracy.
* **Quantization Drop**: Only **1.96%** accuracy drop going from floating-point training to 8-bit integer hardware LUT inference.
* **RTL Verification Parity**: The testbenches verify both designs against 100 test samples and confirm **100% bit-accurate predictions** (zero mismatches) relative to the reference software integer model.

### Advanced Mixed-Precision & Asymptotic Pruning (`[1, 6, 6]` bits)

By running `qat_mixed_demo.jl`, we train a highly optimized mixed-precision KAN using a 1-bit input layer and 6-bit hidden layers and outputs, utilizing an asymptotic pruning schedule (thresholds up to `0.15` for Layer 1 and `1.70` for Layer 2 over 15 epochs) followed by 35 epochs of fine-tuning:

* **Continuous QAT Model Accuracy**: **95.51%**
* **Bit-Accurate Integer LUT Accuracy**: **95.51%**
* **Discretization Loss**: **0.0%** (exactly lossless!)
* **Layer 1 Connection Density**: **18.68%** (a **5.3× active connection reduction**)
* **Layer 2 Connection Density**: **73.91%** (a **1.35× active connection reduction**)
* **LUT Memory Footprint Reduction**:
  * **Layer 1**: $2^8 = 256$ entries reduced to $2^1 = 2$ entries (**128× smaller memory**).
  * **Layer 2**: $2^8 = 256$ entries reduced to $2^6 = 64$ entries (**4× smaller memory**).
* **Tang Nano 20K Fitting**: The combinational adder tree structure fits comfortably within the physical resources of the Sipeed Tang Nano 20K.

---

## How to Run

Ensure you are in the project root directory:
```bash
cd /home/phil/devel/FPGA/KAN_LUT
```

### 1. Train and Discretize the KAN

#### Train 8-bit model:
```bash
julia --project=. examples/MNIST/demo.jl
```

#### Train mixed-precision [1, 6, 6] model:
```bash
julia --project=. examples/MNIST/qat_mixed_demo.jl
```

### 2. Run Custom Inference on Test Digits
After training has generated the JSON files, run the inference script to visualize digits in ASCII and classify them:
```bash
julia --project=. examples/MNIST/inference.jl
```

### 3. Run FPGA Simulations (Verilator)

First, generate the RTL files (for Paradigm 1 targets) and memory files (for Paradigm 2 targets) from the trained JSON:
```bash
cd examples/MNIST/FPGA
make generate
```

#### Run Paradigm 1 (Unrolled Mixed-Precision) Simulation:
```bash
cd examples/MNIST/FPGA
make clean && make run
```
* **Expected Output**: `SUCCESS: All 100 samples verified with 100% bit-accurate parity!`

#### Run Paradigm 1 (Unrolled 8-bit) Simulation:
```bash
cd examples/MNIST/FPGA
make clean && make run_8bit
```
* **Note**: This compilation compiles ~800 verilator-split files to handle the massive unrolled 8-bit ROMs and takes 5-10 minutes.

#### Run Paradigm 2 (Generic BRAM/PSRAM-backed) Simulation:
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

You can launch an interactive desktop application where you draw digits with your mouse on an SDL2 window, and the Verilator simulation of our KAN IP Core runs inference on it in real time.

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

#### Run for Paradigm 1 (Unrolled Mixed-Precision):
```bash
cd examples/MNIST/FPGA
make sim_interactive
```

#### Run for Paradigm 2 (Generic BRAM/PSRAM-backed):
```bash
cd examples/MNIST/FPGA
make -f Makefile.generic sim_interactive
```

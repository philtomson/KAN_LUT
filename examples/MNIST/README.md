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
2. **Paradigm 2 (Parameterized Generic IP Core)**: Located in [hardware/](file:///home/phil/devel/FPGA/KAN_LUT/hardware). A multi-cycle time-multiplexed accelerator that uses dual-port BRAM memory banks and can reload weights at run-time.

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
The raw weight data for the `196 -> 32 -> 10` network is exactly **23.5 Megabits**:
* **Layer 1**: $196 \text{ inputs} \times 32 \text{ outputs} \times 256 \text{ entries} \times 14 \text{ bits/entry} \approx 22.48\text{ Mbits}$.
* **Layer 2**: $32 \text{ inputs} \times 10 \text{ outputs} \times 256 \text{ entries} \times 13 \text{ bits/entry} \approx 1.06\text{ Mbits}$.

* In **Paradigm 1**, weights are hardcoded as constant ROM lookups. The compiler (e.g., Yosys or Vivado) applies **Boolean minimization** and **logic pruning**, removing zero-weights and matching duplicate paths. This dramatically reduces the actual logic footprint.
* In **Paradigm 2**, the memory banks are writeable, allowing model reloading at boot time. Because weights can change, the compiler **cannot** prune the storage. It must allocate the full physical BRAM capacity (~23.5 Mbits), which makes the design light on logic gates but memory-intensive.

### Classification Accuracy

Both Verilog implementations achieve the **exact same classification accuracy** of **92.30%** (on the full MNIST test set), as they implement identical quantized mathematical logic:
* **Continuous Float KAN Model**: **94.56%** accuracy.
* **Bit-Accurate Verilog (Paradigm 1 & 2)**: **92.30%** accuracy.
* **Quantization Drop**: Only **2.26%** accuracy drop going from floating-point training to 8-bit integer hardware LUT inference. This drop is minimized by using a domain range of $[-8.0, 8.0]$ to prevent activation clipping in the inter-layer adder tree.
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
julia --project=. hardware/generate_mem_files.jl

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
cd ../../../hardware
make clean && make run
```
* **Expected Output**: `SUCCESS: All 100 samples verified with 100% bit-accurate parity on Generic KAN Core!`

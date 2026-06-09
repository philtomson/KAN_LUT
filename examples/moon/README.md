# Moons Classification Demo

This directory contains the synthetic 2D interleaving half-moons classification demo, showcasing the end-to-end KAN-LUT pipeline.

## Files

1. **`demo.jl`**:
   * Generates the moons dataset.
   * Trains a 2-layer KAN (`2 -> 8 -> 2`) on GPU (`AMDGPU.jl` or CPU backup).
   * Generates static 8-bit integer L-LUT files (`moons_luts_layer1.json` and `moons_luts_layer2.json`) in this directory.
   * Performs simulated on-device online learning (ICML '26 style) under a 40-degree distribution shift.
2. **`visualize.jl`**:
   * Trains the model, performs quantization, and runs online adaptation.
   * Generates a 2D meshgrid to compute decision boundaries.
   * Saves a 2x2 comparison image as `moons_plots.png` in this directory.
3. **`inference.jl`**:
   * Demonstrates how to load the static JSON LUT files.
   * Feeds arbitrary raw floating-point input coordinates.
   * Quantizes the inputs, performs bit-accurate fixed-point lookup/accumulation, and decodes the predicted classes.

---

## How to Run

Ensure you are in the project root directory:
```bash
cd /home/phil/devel/FPGA/KAN_LUT
```

### 1. Run the Training and Quantization Demo
This script will output the training process, generate the static L-LUTs, and output online adaptation logs:
```bash
julia --project=. examples/moon/demo.jl
```

### 2. Run the Custom Inference Script
After running `demo.jl`, you can load the saved LUTs and perform inference on custom coordinates:
```bash
julia --project=. examples/moon/inference.jl
```

### 3. Generate the Decision Boundary Plots
This script will produce a 2x2 grid plot (`moons_plots.png`) comparing the continuous KAN, quantized LUT, pre-adapted shift, and post-adapted shift states:
```bash
julia --project=. examples/moon/visualize.jl
```

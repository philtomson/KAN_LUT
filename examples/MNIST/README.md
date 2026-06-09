# MNIST Digits Classification Demo

This directory implements KAN training, 8-bit L-LUT discretization, and bit-accurate inference on the MNIST handwritten digit classification task.

To make the KAN extremely lightweight and fast to train, 28x28 pixel images are downsampled to 14x14 pixel images ($196$ input features) using $2 \times 2$ average pooling.

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

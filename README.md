# Kolmogorov-Arnold Network (KAN) LUT FPGA Framework

Kolmogorov-Arnold Networks (KANs) replace the fixed node activation functions of Multi-Layer Perceptrons (MLPs) with learnable univariate activation functions on the edges (connections). By representing these functions using B-splines, KANs achieve comparable or superior accuracy to MLPs with significantly fewer parameters. This edge-centric, univariate formulation makes KANs exceptionally well-suited for FPGA implementations: instead of executing resource-heavy floating-point multiplications or complex transcendental functions, each edge function can be discretized and mapped directly to a hardware Look-Up Table (L-LUT). Inference is reduced to highly parallel, low-latency LUT lookups followed by simple integer adder trees, maximizing hardware resource efficiency and throughput.

This Julia framework implements KANs optimized for resource-efficient FPGA deployment. It supports GPU-accelerated training using `Flux.jl` and `AMDGPU.jl`, static quantization to 8-bit integer Lookup Tables (L-LUTs), and simulated FPGA bit-accurate inference and online learning.

## Key Features

1. **GPU-Accelerated B-spline Engine**: Fully vectorized linear B-spline evaluation supporting seamless execution on AMD GPUs.
2. **Flux-Compatible KAN Layer**: Differentiable `KANLayer` designed for standard Flux workflows and gradient backpropagation.
3. **Hardware-Optimized Quantization**: Discretization logic that converts continuous activations into static 8-bit L-LUT arrays. It bakes the input offset shift ($a / d_{in}$) directly into the tables, saving hardware subtractor stages in the adder tree.
4. **Bit-Accurate FPGA Simulator**: Integer inference engines simulating FPGA adder-tree accumulations, bit-shifts, and saturation clipping.
5. **Dynamic Online Learning**: Precomputes B-spline basis function LUTs allowing real-time, sparse coefficient updates on-device during distribution shifts.

---

## Installation & Setup

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone git@github.com:philtomson/KAN_LUT.git
   cd KAN_LUT
   ```
2. Start Julia and instantiate the project dependencies:
   ```bash
   julia --project=. -e 'using Pkg; Pkg.instantiate()'
   ```

---

## Available Examples

We provide two end-to-end examples showcasing training, quantization, and verification:

1. **Moons Classification (`examples/moon/`)**:
   * Synthetic 2D half-moons binary classification.
   * Includes boundary visualization plots comparing continuous, quantized, and adapted models.
   * See the [Moons README](file:///home/phil/devel/FPGA/KAN_LUT/examples/moon/README.md) for details.
2. **MNIST Digits Classification (`examples/MNIST/`)**:
   * The classic handwritten digits classification task downsampled to 14x14 pixels.
   * Prints digit ASCII art inside the terminal and verifies bit-accurate inference.
   * See the [MNIST README](file:///home/phil/devel/FPGA/KAN_LUT/examples/MNIST/README.md) for details.

---

## Running the Moons Demo

To run the end-to-end demo which trains the KAN on an AMD GPU, quantizes it to 8-bit static LUTs, and runs online adaptation:
```bash
julia --project=. examples/moon/demo.jl
```

### Expected Output
```text
==================================================
STEP 1: Generating Moons Classification Dataset
==================================================
Train features size: (2, 400)
Test features size: (2, 100)
Train input X range: [-1.1832741, 2.2902308]

==================================================
STEP 2: Defining and Training KAN on GPU
==================================================
Training device: AMD GPU
Epoch 1: Loss = 0.5749, Accuracy = 80.5%
Epoch 25: Loss = 0.0133, Accuracy = 100.0%
...
Final Continuous Model Accuracy: 100.0%

==================================================
STEP 3: Static LUT Discretization (KANELÉ Style)
==================================================
Generating L-LUTs for Layer 1...
Generating L-LUTs for Layer 2...
Saved Layer 1 LUTs to: examples/moon/moons_luts_layer1.json
Saved Layer 2 LUTs to: examples/moon/moons_luts_layer2.json

==================================================
STEP 4: Bit-Accurate Fixed-Point Inference
==================================================
Bit-Accurate Integer LUT Accuracy: 100.0%
Accuracy Drop: 0.0%

==================================================
STEP 5: Online Learning Adaptation Simulation
==================================================
Accuracy on shifted dataset (Pre-adaptation): 60.67%
Simulating online learning loop for 200 samples...
  Step 1: Online Sample Loss = 5.45
  ...
  Step 200: Online Sample Loss = 0.4242
Accuracy on shifted dataset (Post-adaptation): 72.67%
Adaptation Gain: +12.0%
```

---

## Running Custom Inference

After running the demo, the trained LUT arrays are saved as JSON files in `examples/moon/`. You can load these tables and run custom bit-accurate inference on arbitrary 2D coordinate points using the inference script:
```bash
julia --project=. examples/moon/inference.jl
```

---

## Visualizing the Moons Demo

To visually inspect the decision boundaries of the trained KAN models (continuous model, integer LUT model, and adapted online model), we provide a visualization script.

### 1. Run the Visualization Script
Execute the script using Julia:
```bash
julia --project=. examples/moon/visualize.jl
```
This script will:
* Install the `Plots` package if it is not already installed.
* Train a KAN model, discretize it, and perform the distribution shift/online adaptation.
* Generate a grid of points to evaluate the decision boundary at each stage.
* Save the comparison plot as a PNG image: `examples/moon/moons_plots.png`.

### 2. Output Plot Layout
The saved image `examples/moon/moons_plots.png` contains four subplots:
1. **Continuous Model Decision Boundary**: Displays the trained KAN's highly non-linear boundary separating the two half-moons.
2. **Bit-Accurate Integer LUT Boundary**: Shows the decision boundary generated by the static 8-bit quantized integer LUT simulation, confirming zero visual degradation compared to the continuous reference.
3. **Shifted Dataset (Pre-Adaptation)**: Illustrates the rotated dataset overlaid on the old decision boundary, highlighting how the distribution shift degrades model accuracy.
4. **Shifted Dataset (Post-Adaptation)**: Shows the adjusted decision boundary after sparse online B-spline updates, showing how the local basis updates adapted the model to the shift.

---

## References

* **KAN-FPGA Post**: [aarushgupta.io/posts/kan-fpga](https://aarushgupta.io/posts/kan-fpga/)


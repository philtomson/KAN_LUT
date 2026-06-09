# Kolmogorov-Arnold Network (KAN) LUT implementation in Julia

This plan proposes a comprehensive Julia implementation of Kolmogorov-Arnold Networks (KANs) targeting FPGA hardware acceleration, combining the core designs of two recent publications:
1. **KANELÉ (FPGA '26)**: Static LUT-based inference where entire pre-trained edge activations $\phi_{q,p}(x)$ are mapped to Look-up Tables (L-LUTs).
2. **Online KAN Learning (ICML '26)**: On-FPGA training and inference where the B-spline basis functions $\{B_i\}$ are precomputed in LUTs, and the coefficients $c_{q,p,i}$ are stored in memory and updated dynamically in real-time.

Training and simulation will be GPU-accelerated using AMD GPUs via `AMDGPU.jl` and `Flux.jl`.

## Proposed Changes

We will create a new Julia package structure inside `/home/phil/devel/FPGA/KAN_LUT`.

### Julia Codebase Structure
The package will contain the following files:

#### [NEW] Project.toml
Defines the Julia project and its dependencies: `Flux`, `AMDGPU`, `LinearAlgebra`, `Random`, `JSON`.

#### [NEW] src/KAN_LUT.jl
Entry point of the package. Exports types, layer structures, training utilities, and inference functions.

#### [NEW] src/bspline.jl
Implements B-spline evaluation:
- Fast vectorized linear spline (degree 1) evaluation for training and CPU/GPU execution.
- Cox-de Boor recursion for arbitrary-degree reference.
- Precomputed B-spline basis LUT generation for online learning.

#### [NEW] src/lut_generation.jl
Logic for discretizing KAN layers:
- **Static Mode**: Evaluates and quantizes the entire activation function $\phi_{q,p}(x)$ to create direct L-LUTs.
- **Dynamic Mode**: Precomputes and quantizes the B-spline basis functions $\{B_i\}$ and their derivatives into LUTs, leaving coefficients as trainable parameters.

#### [NEW] src/inference.jl
Implements the fast inference engines:
- `float_lut_inference`: CPU inference with Floating-Point LUTs.
- `fixed_lut_inference`: Bit-accurate fixed-point integer inference using shifted integer LUTs, simulating hardware adder trees and saturation arithmetic.
- `online_lut_inference`: Simulates the dynamic online-learning forward pass, where B-spline basis values are looked up from basis LUTs and scaled by the current coefficients.

#### [NEW] src/utils.jl
Helper functions for quantization, clipping, and importing/exporting model parameters (e.g., from/to JSON).

#### [NEW] test/runtests.jl
Unit tests verifying the B-spline basis functions, GPU execution compatibility, quantization correctness, and bit-accurate inference.

#### [NEW] examples/demo.jl
A runnable demo demonstrating a complete end-to-end workflow on the **Moons Classification** dataset:
1. **Data Generation**: Generate synthetic 2D non-linear classification data (interleaving half-moons).
2. **Model Definition**: Instantiate a `2 -> 8 -> 2` KAN with 1D linear splines (degree 1) over grid range $[-2.0, 2.0]$ and grid size $G=10$.
3. **GPU Training**: Port model and data to AMD GPU using `AMDGPU.jl` and train using `Flux` (cross-entropy loss + AdamW optimizer) for 100 epochs, printing the loss reduction.
4. **Static Discretization**: Export the trained model to static 8-bit integer L-LUTs.
5. **Bit-Accurate Simulation**: Run the integer-based `fixed_lut_inference` on test data, demonstrating that the discrete model achieves accuracy close to the continuous floating-point model.
6. **Online Learning Simulation**: Introduce a distribution shift (e.g., rotating the moons), and show how the model adapts in real-time by running online coefficient updates using B-spline basis lookup tables.

---

## Detailed Design Specifications

### 1. Static Mode (KANELÉ style)
For static, pre-trained models, the continuous activation functions are fixed.
- **LUT Size**: For $n_{in}$-bit input, the L-LUT contains $2^{n_{in}}$ entries.
- **Discretization**:
  $$\text{LUT}_{q,p}[v] = \text{round}\left( \phi_{q,p}(a + v \cdot \delta_{in}) \cdot \frac{2^{n_{out}}-1}{b-a} \cdot 2^{k} \right)$$
- **Execution**: Node outputs are computed as simple integer sums of LUT values, followed by a shift-and-clip operation:
  $$v_q = \text{clip}\left( \text{round}\left(\frac{\sum_p \text{LUT}_{q,p}[v_p]}{2^k}\right), 0, 2^{n_{out}} - 1 \right)$$

### 2. Dynamic Online-Learning Mode (ICML '26 style)
For models training in real time, the coefficients $c$ change continuously.
- **Basis LUT**: Since the B-spline basis functions $\{B_i\}$ are fixed, we precompute their values over a fine grid (e.g. 8-bit offset resolution).
- **Locality Optimization**:
  For degree $S$, only $S+1$ basis functions are non-zero. For an input $x$, we compute:
  - Interval Index: $j = \lfloor (x - a) / h \rfloor$
  - Offset: $x_{offset} = (x - a) \pmod h$
- **Forward Pass**:
  1. Retrieve $S+1$ basis values from the Basis LUT using $x_{offset}$.
  2. Retrieve the $S+1$ active coefficients $\{c_{q,p,j+m}\}_{m=1}^{S+1}$.
  3. Compute $\phi_{q,p}(x) = \sum_{m=1}^{S+1} c_{q,p,j+m} B_m(x_{offset})$.
- **Backward Pass**:
  - Precompute the derivatives $\frac{\text{d}B_m}{\text{d}x}$ in a separate LUT.
  - The coefficient gradient is $\frac{\partial \mathcal{L}}{\partial c_{q,p,j+m}} = \frac{\partial \mathcal{L}}{\partial y_q} B_m(x_{offset})$.
  - Since only $S+1$ coefficients are active per input, the gradient update is extremely sparse.

---

## Verification Plan

### Automated Tests
We will write a test suite in `test/runtests.jl` to:
1. Verify `AMDGPU` execution for both static and dynamic KAN layers.
2. Confirm the B-spline basis functions and derivatives evaluate correctly.
3. Validate that both static L-LUT and dynamic basis-LUT inference match continuous outputs.

### Manual Verification
- Run `examples/demo.jl` to perform a full lifecycle simulation: training, quantization, static L-LUT export, and online coefficient update simulation.

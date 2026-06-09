# examples/moon/inference.jl
# Demonstrates loading trained/discretized KAN LUTs from JSON and running 
# bit-accurate fixed-point inference in Julia.

using KAN_LUT
using JSON

println("==================================================")
println("STEP 1: Loading Trained KAN LUTs from JSON")
println("==================================================")

# Construct paths relative to this script's directory
lut_path1 = joinpath(@__DIR__, "moons_luts_layer1.json")
lut_path2 = joinpath(@__DIR__, "moons_luts_layer2.json")

if !isfile(lut_path1) || !isfile(lut_path2)
    error("Trained LUT files not found!\nPlease run `julia --project=. examples/moon/demo.jl` first to train the model and save the LUTs.")
end

# Load LUT arrays and quantization parameters for both layers
luts1, a1, b1, n_in1, n_out1, k1 = load_lut_json(lut_path1)
luts2, a2, b2, n_in2, n_out2, k2 = load_lut_json(lut_path2)

println("Layer 1 Loaded LUT shape: ", size(luts1))
println("  - Input range (a, b): [", a1, ", ", b1, "]")
println("  - Bit-widths (n_in, n_out): (", n_in1, ", ", n_out1, ")")
println("  - Fractional bits (k): ", k1)

println("\nLayer 2 Loaded LUT shape: ", size(luts2))
println("  - Input range (a, b): [", a2, ", ", b2, "]")
println("  - Bit-widths (n_in, n_out): (", n_in2, ", ", n_out2, ")")
println("  - Fractional bits (k): ", k2)

println("\n==================================================")
println("STEP 2: Defining Sample Input Points (Moons Class 0 & Class 1)")
println("==================================================")

# Two sample coordinates:
# Sample 1: Class 0 typical point [0.5, 0.8]
# Sample 2: Class 1 typical point [1.5, -0.3]
X_samples = Float32[
    0.5  1.5;
    0.8 -0.3
]
println("Sample coordinates:\n", X_samples)

println("\n==================================================")
println("STEP 3: Running Bit-Accurate Fixed-Point Inference")
println("==================================================")

# 1. Quantize continuous input samples to n_in-bit integers mapping to [0, 2^n_in - 1]
X_int = quantize_input(X_samples, a1, b1, n_in1)
println("Quantized Layer 1 inputs:\n", X_int)

# 2. Run Layer 1 bit-accurate fixed-point inference (outputs integers in 0:2^n_out1-1)
H_int = fixed_lut_inference(luts1, X_int, n_out1, k1)
println("\nLayer 1 integer output activations:\n", H_int)

# 3. Run Layer 2 bit-accurate fixed-point inference (outputs integers in 0:2^n_out2-1)
Y_int = fixed_lut_inference(luts2, H_int, n_out2, k2)
println("\nLayer 2 raw integer output scores:\n", Y_int)

# 4. Dequantize the output integer scores back to continuous float values for interpretation
Y_dequant = dequantize_output(Y_int, a2, b2, n_out2)
println("\nDequantized real-valued output predictions (Logits):\n", Y_dequant)

# 5. Extract class predictions using argmax
for idx in 1:size(X_samples, 2)
    pred_class = argmax(Y_dequant[:, idx]) - 1 # Convert 1-based index to 0-based class
    val_class = pred_class == 0 ? "Class 0 (Outer Moon)" : "Class 1 (Inner Moon)"
    println("Sample $idx: Coordinate $(X_samples[:, idx]) -> Predicted: $val_class")
end
println("\nINFERENCE COMPLETED SUCCESSFULLY!")
println("==================================================")

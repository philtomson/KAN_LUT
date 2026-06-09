# examples/MNIST/inference.jl
# Demonstrates loading trained MNIST KAN LUTs and running bit-accurate fixed-point
# inference on actual digits from the test set, displaying them in ASCII art.

using KAN_LUT
using JSON
using MLDatasets

# Helper function to downsample 28x28 MNIST images to 14x14
function downsample_mnist(X::AbstractMatrix{Float32})
    out = zeros(Float32, 14, 14)
    for i in 1:14
        for j in 1:14
            out[i, j] = (
                X[2*i-1, 2*j-1] + 
                X[2*i-1, 2*j]   + 
                X[2*i,   2*j-1] + 
                X[2*i,   2*j]
            ) / 4.0f0
        end
    end
    return out
end

# Prints a 14x14 downsampled image in the terminal using ASCII characters
function print_digit_ascii(img::AbstractMatrix{Float32})
    # We transpose / index correctly so it displays right-side up
    # MNIST features are (height, width) but we read column-major.
    # Transposing to print row-by-row:
    chars = [" ", ".", ":", "-", "=", "+", "*", "#", "%", "@"]
    for y in 1:14
        for x in 1:14
            val = img[x, y]
            idx = clamp(floor(Int, val * 9) + 1, 1, 10)
            print(chars[idx], " ")
        end
        println()
    end
end

println("==================================================")
println("STEP 1: Loading Trained MNIST LUTs from JSON")
println("==================================================")

lut_path1 = joinpath(@__DIR__, "mnist_luts_layer1.json")
lut_path2 = joinpath(@__DIR__, "mnist_luts_layer2.json")

if !isfile(lut_path1) || !isfile(lut_path2)
    error("Trained LUT files not found!\nPlease run `julia --project=. examples/MNIST/demo.jl` first to train the model and save the LUTs.")
end

luts1, a1, b1, n_in1, n_out1, k1 = load_lut_json(lut_path1)
luts2, a2, b2, n_in2, n_out2, k2 = load_lut_json(lut_path2)

println("Layer 1 Loaded LUT shape: ", size(luts1))
println("Layer 2 Loaded LUT shape: ", size(luts2))

println("\n==================================================")
println("STEP 2: Fetching MNIST Test Samples")
println("==================================================")

# Load test dataset
test_data = MNIST(split=:test)

# Pick 3 random indices to run inference on
sample_indices = [12, 42, 101] # Index 12 is a 9, index 42 is a 4, index 101 is a 6 (standard MNIST indices)

for (idx_run, test_idx) in enumerate(sample_indices)
    raw_img = test_data.features[:, :, test_idx]
    true_label = test_data.targets[test_idx]
    
    println("\n--------------------------------------------------")
    println("Sample $idx_run: Test Image Index #$test_idx (True Label: $true_label)")
    println("--------------------------------------------------")
    
    # Downsample and display
    down_img = downsample_mnist(raw_img)
    print_digit_ascii(down_img)
    
    # Flatten & Quantize
    x_flat = reshape(down_img, 196, 1) # Shape (196, 1)
    x_int = quantize_input(x_flat, a1, b1, n_in1)
    
    # Bit-accurate inference
    h_int = fixed_lut_inference(luts1, x_int, n_out1, k1)
    y_int = fixed_lut_inference(luts2, h_int, n_out2, k2)
    
    # Dequantize predictions to read logits
    y_de = dequantize_output(y_int, a2, b2, n_out2)
    pred_digit = argmax(y_de[:, 1]) - 1
    
    println("\nPredicted Digit: $pred_digit (Confidence/Logits: $(round.(y_de[:, 1], digits=2)))")
    if pred_digit == true_label
        println("Result: SUCCESS! Match!")
    else
        println("Result: MISMATCH!")
    end
end

println("\n==================================================")
println("INFERENCE COMPLETED SUCCESSFULLY!")
println("==================================================")

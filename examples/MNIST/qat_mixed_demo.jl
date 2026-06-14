# examples/MNIST/qat_mixed_demo.jl
# Trains a mixed-precision QAT KAN model on downsampled MNIST digits (14x14 pixels) on GPU,
# performs connection pruning, discretizes it to mixed-bitwidth QAT-LUT JSON files [1, 6, 6] bits,
# and evaluates the bit-accurate integer accuracy.

using KAN_LUT
using Flux
using AMDGPU
using CUDA
using LinearAlgebra
using Random
using JSON
using MLDatasets

# Set seed for reproducibility
Random.seed!(42)

# Helper function to downsample 28x28 MNIST images to 14x14 via 2x2 average pooling
function downsample_mnist(X::AbstractArray{Float32, 3})
    h, w, n = size(X)
    out = zeros(Float32, 14, 14, n)
    for idx in 1:n
        for i in 1:14
            for j in 1:14
                out[i, j, idx] = (
                    X[2*i-1, 2*j-1, idx] + 
                    X[2*i-1, 2*j,   idx] + 
                    X[2*i,   2*j-1, idx] + 
                    X[2*i,   2*j,   idx]
                ) / 4.0f0
            end
        end
    end
    return out
end

println("==================================================")
println("STEP 1: Loading and Downsampling MNIST Dataset")
println("==================================================")
# Load MNIST dataset from MLDatasets
train_data = MNIST(split=:train)
test_data = MNIST(split=:test)

println("Downsampling train images from 28x28 to 14x14...")
X_train_down = downsample_mnist(train_data.features)
X_test_down = downsample_mnist(test_data.features)

# Flatten to (196, N)
X_train = reshape(X_train_down, 196, :)
X_test = reshape(X_test_down, 196, :)

# One-hot encode targets (0:9)
Y_train = Float32.(Flux.onehotbatch(train_data.targets, 0:9))
Y_test = Float32.(Flux.onehotbatch(test_data.targets, 0:9))

println("Train features size: ", size(X_train))
println("Test features size:  ", size(X_test))
println("Train input pixel range: [", minimum(X_train), ", ", maximum(X_train), "]")

println("\n==================================================")
println("STEP 2: Defining and Training Mixed-Precision QAT KAN on GPU")
println("==================================================")
nbits_in = 1
nbits_hidden = 6

# Mixed bit-width model: 1-bit input layer, 6-bit hidden layers/outputs
input_layer = QATInputLayer(196, nbits=nbits_in)
layer1 = QATKANLayer(196, 64, G=5, a=-8.0, b=8.0, nbits=nbits_hidden)
layer2 = QATKANLayer(64, 10, G=5, a=-8.0, b=8.0, nbits=nbits_hidden)
model = Chain(input_layer, layer1, layer2)

# Select GPU if available
device, device_name = if CUDA.functional()
    cu, "CUDA GPU"
elseif AMDGPU.functional()
    roc, "AMD GPU"
else
    cpu, "CPU"
end
println("Training device: ", device_name)

# Move model and data to device
model = fmap(device, model)
X_train_dev = device(X_train)
Y_train_dev = device(Y_train)

# Setup optimizer
opt_state = Flux.setup(Flux.AdamW(0.01f0), model)

batch_size = 1000
n_samples = size(X_train, 2)
num_epochs = 50

# Training loop
for epoch in 1:num_epochs
    global model, opt_state
    # Shuffle indices
    indices = randperm(n_samples)
    epoch_loss = 0.0f0
    num_batches = 0
    
    for i in 1:batch_size:n_samples
        idx = indices[i:min(i+batch_size-1, n_samples)]
        bx = X_train_dev[:, idx]
        by = Y_train_dev[:, idx]
        
        # Forward pass and gradient computation
        loss, grads = Flux.withgradient(model) do m
            logits = m(bx)
            Flux.Losses.logitcrossentropy(logits, by)
        end
        
        Flux.update!(opt_state, model, grads[1])
        epoch_loss += loss
        num_batches += 1
    end
    
    avg_loss = epoch_loss / num_batches
    
    # Calculate test accuracy every 5 epochs
    if epoch % 5 == 0 || epoch == 1
        preds = Array(model(device(X_test)))
        acc = sum(argmax(preds, dims=1) .== argmax(Y_test, dims=1)) / size(Y_test, 2)
        println("Epoch $(epoch): Avg Loss = $(round(avg_loss, digits=4)), Test Accuracy = $(round(acc * 100, digits=2))%")
    end
    
    # Trigger Asymptotic Connection Pruning starting at Epoch 6
    max_thresh_l1 = 0.15f0
    max_thresh_l2 = 1.70f0
    warmup_epochs = 5
    target_epoch = 15
    
    if epoch > warmup_epochs && epoch <= target_epoch
        t = epoch - warmup_epochs
        denom = max(target_epoch - warmup_epochs, 1)
        k = log(20.0f0) / denom
        scale_factor = 1.0f0 - exp(-k * t)
        
        current_threshold_l1 = min(max_thresh_l1 * scale_factor, max_thresh_l1)
        current_threshold_l2 = min(max_thresh_l2 * scale_factor, max_thresh_l2)
        
        # Move model to CPU to compute spline norms and apply pruning selector
        model_cpu = fmap(cpu, model)
        il = model_cpu[1]
        l1 = model_cpu[2]
        l2 = model_cpu[3]
        
        # Get state space of previous outputs for forward/backward norm calculation
        state_space_in = get_state_space(il.scale, nbits_in)
        state_space_l1 = get_state_space(l1.scale, nbits_hidden)
        
        println("\n--- Triggering Asymptotic Connection Pruning (Epoch $(epoch), Thresh L1 = $(round(current_threshold_l1, digits=4)), Thresh L2 = $(round(current_threshold_l2, digits=4))) ---")
        
        # Prune Layer 2 (no subsequent layer) using current_threshold_l2
        density2 = prune_layer!(l2, current_threshold_l2, nothing, state_space_l1)
        # Prune Layer 1 (backward prune using Layer 2 selector) using current_threshold_l1
        density1 = prune_layer!(l1, current_threshold_l1, l2.selector, state_space_in)
        
        println("Layer 1 active connections density: $(round(density1 * 100, digits=2))%")
        println("Layer 2 active connections density: $(round(density2 * 100, digits=2))%")
        
        # Copy the updated selector masks and weights in-place back to the training device (no fmap, preserving optimizer state)
        copyto!(model[2].w_base, model_cpu[2].w_base)
        copyto!(model[2].w_spline, model_cpu[2].w_spline)
        copyto!(model[2].selector, model_cpu[2].selector)
        
        copyto!(model[3].w_base, model_cpu[3].w_base)
        copyto!(model[3].w_spline, model_cpu[3].w_spline)
        copyto!(model[3].selector, model_cpu[3].selector)
        
        println("Pruned weights and selectors copied to $(device_name) in-place (optimizer state preserved).\n")
    end
    
    # Clean up GPU memory pool after each epoch to prevent VRAM bloat
    GC.gc(true)
    if device_name == "AMD GPU"
        AMDGPU.HIP.reclaim()
    end
end

# Move back to CPU for discretization and validation
model_cpu = fmap(cpu, model)
il = model_cpu[1]
l1 = model_cpu[2]
l2 = model_cpu[3]

# Measure final continuous accuracy on CPU
continuous_preds = model_cpu(X_test)
continuous_acc = sum(argmax(continuous_preds, dims=1) .== argmax(Y_test, dims=1)) / size(Y_test, 2)
println("\nFinal Continuous QAT Model Test Accuracy: $(round(continuous_acc * 100, digits=2))%")

println("\n==================================================")
println("STEP 3: QAT LUT Discretization (Mixed-Bitwidth [1, 6, 6])")
println("==================================================")

println("Generating QAT-LUTs for Layer 1...")
luts1 = generate_qat_luts(l1, il.scale[1], in_nbits=nbits_in)
println("Generating QAT-LUTs for Layer 2...")
luts2 = generate_qat_luts(l2, l1.scale[1], in_nbits=nbits_hidden)

println("Layer 1 QAT-LUT array shape (num_entries, d_out, d_in): ", size(luts1))
println("Layer 2 QAT-LUT array shape (num_entries, d_out, d_in): ", size(luts2))

# Define helper to save QAT LUTs with float scales
function save_qat_lut_json(filename::String, luts::Array{Int32, 3}, layer::QATKANLayer, input_scale::Real, in_nbits::Int)
    num_entries, d_out, d_in = size(luts)
    
    lut_dict = Dict{String, Any}()
    
    # Store QAT metadata
    lut_dict["metadata"] = Dict(
        "a" => Float64(layer.a),
        "b" => Float64(layer.b),
        "nbits" => Int(layer.nbits),
        "in_nbits" => Int(in_nbits),
        "input_scale" => Float64(input_scale),
        "output_scale" => Float64(layer.scale[1]),
        "d_in" => Int(d_in),
        "d_out" => Int(d_out)
    )
    
    # Store LUT tables for each edge (q, p)
    luts_data = Dict{String, Vector{Int32}}()
    for p in 1:d_in
        for q in 1:d_out
            key = "lut_$(q)_$(p)"
            luts_data[key] = luts[:, q, p]
        end
    end
    lut_dict["luts"] = luts_data
    
    open(filename, "w") do io
        JSON.print(io, lut_dict, 2)
    end
end

model_path1 = joinpath(@__DIR__, "mnist_qat_luts_mixed_layer1.json")
model_path2 = joinpath(@__DIR__, "mnist_qat_luts_mixed_layer2.json")
save_qat_lut_json(model_path1, luts1, l1, il.scale[1], nbits_in)
save_qat_lut_json(model_path2, luts2, l2, l1.scale[1], nbits_hidden)

println("Saved Layer 1 QAT-LUTs to: ", model_path1)
println("Saved Layer 2 QAT-LUTs to: ", model_path2)

println("\n==================================================")
println("STEP 4: Fixed-Point Inference Validation")
println("==================================================")

# Quantize the test inputs (1-bit signed values: {-1, 1})
x_trans = il.bn(X_test) .+ il.bias
x_clamped = clamp.(x_trans, -1.0f0, 1.0f0)
# For 1-bit, the output activations are sign of input times scale.
# In the integer domain, we represent the sign as {-1, 1}.
x_int = map(v -> v < 0.0f0 ? Int32(-1) : Int32(1), x_clamped)

# Map {-1, 1} to unsigned indices {0, 1} for first layer lookup
x_uint = div.(x_int .+ 1, 2)

# Run bit-accurate integer LUT inference simulation
# First layer maps unsigned 1-bit inputs to 6-bit signed activations
y_int_l1 = qat_fixed_lut_inference(luts1, x_uint, nbits_hidden, in_nbits=nbits_in, is_first_layer=true)
# Second layer maps signed 6-bit activations to 6-bit signed outputs
y_int_l2 = qat_fixed_lut_inference(luts2, y_int_l1, nbits_hidden, in_nbits=nbits_hidden, is_first_layer=false)

# Evaluate bit-accurate simulation accuracy
lut_acc = sum(argmax(y_int_l2, dims=1) .== argmax(Y_test, dims=1)) / size(Y_test, 2)
println("Bit-Accurate Integer QAT-LUT Test Accuracy: $(round(lut_acc * 100, digits=2))%")
println("Accuracy Drop vs. Continuous: $(round((continuous_acc - lut_acc) * 100, digits=2))%")

# Write first 100 test vectors to FPGA folder for SystemVerilog testbench validation
tb_data_path = joinpath(@__DIR__, "FPGA", "tb_data_mixed.txt")
open(tb_data_path, "w") do io
    for b in 1:100
        # Write inputs (196 values as 2-digit hex: 00 or 01)
        for p in 1:196
            val = x_uint[p, b]
            write(io, lowercase(string(val, base=16, pad=2)))
        end
        write(io, " ")
        # Write outputs (10 values as 2-digit hex, masking to 6 bits: val & 0x3f)
        for q in 1:10
            val = y_int_l2[q, b]
            val_masked = Int(val) & 0x3f
            write(io, lowercase(string(val_masked, base=16, pad=2)))
        end
        write(io, "\n")
    end
end
println("Saved 100 RTL test vectors to: ", tb_data_path)

println("\n==================================================")
println("MNIST QAT MIXED DEMO COMPLETED SUCCESSFULLY!")
println("==================================================")

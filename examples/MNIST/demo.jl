# examples/MNIST/demo.jl
# Trains a KAN model on downsampled MNIST digits (14x14 pixels) on GPU,
# discretizes it to 8-bit L-LUT JSON files, and evaluates the bit-accurate integer accuracy.

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
println("STEP 2: Defining and Training KAN on GPU")
println("==================================================")
# We define a 2-layer KAN:
# Both layers must share matching domain boundaries to prevent scale/offset mismatch
# in the direct cascaded integer LUT lookups. We widen the domain to [-8.0, 8.0]
# to prevent hidden layer activation clipping.
layer1 = KANLayer(196, 64, G=5, a=-8.0, b=8.0)
layer2 = KANLayer(64, 10, G=5, a=-8.0, b=8.0)
model = Chain(layer1, layer2)

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

# Training loop (15 epochs)
for epoch in 1:15
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
end

# Move back to CPU for discretization
model_cpu = fmap(cpu, model)
l1 = model_cpu[1]
l2 = model_cpu[2]

# Measure final continuous accuracy on CPU
continuous_preds = model_cpu(X_test)
continuous_acc = sum(argmax(continuous_preds, dims=1) .== argmax(Y_test, dims=1)) / size(Y_test, 2)
println("\nFinal Continuous Model Test Accuracy: $(round(continuous_acc * 100, digits=2))%")

println("\n==================================================")
println("STEP 3: Static LUT Discretization (8-bit)")
println("==================================================")
n_in = 8
n_out = 8
k = 4

println("Generating L-LUTs for Layer 1...")
luts1 = generate_static_luts(l1, n_in, n_out, k)
println("Generating L-LUTs for Layer 2...")
luts2 = generate_static_luts(l2, n_in, n_out, k)

println("Layer 1 LUT array shape: ", size(luts1))
println("Layer 2 LUT array shape: ", size(luts2))

# Save LUTs to JSON
model_path1 = joinpath(@__DIR__, "mnist_luts_layer1.json")
model_path2 = joinpath(@__DIR__, "mnist_luts_layer2.json")
save_lut_json(model_path1, luts1, l1.a, l1.b, n_in, n_out, k)
save_lut_json(model_path2, luts2, l2.a, l2.b, n_in, n_out, k)
println("Saved Layer 1 LUTs to: ", model_path1)
println("Saved Layer 2 LUTs to: ", model_path2)

println("\n==================================================")
println("STEP 4: Bit-Accurate Fixed-Point Inference Validation")
println("==================================================")
# Quantize the test inputs
X_test_int = quantize_input(X_test, l1.a, l1.b, n_in)

# Run bit-accurate integer LUT inference (adder tree + clipping)
H_int = fixed_lut_inference(luts1, X_test_int, n_out, k)
Y_pred_int = fixed_lut_inference(luts2, H_int, n_out, k)

# Evaluate accuracy
lut_acc = sum(argmax(Y_pred_int, dims=1) .== argmax(Y_test, dims=1)) / size(Y_test, 2)
println("Bit-Accurate Integer LUT Test Accuracy: $(round(lut_acc * 100, digits=2))%")
println("Accuracy Drop: $(round((continuous_acc - lut_acc) * 100, digits=2))% (typically < 2%)")

println("\n==================================================")
println("MNIST DEMO COMPLETED SUCCESSFULLY!")
println("==================================================")

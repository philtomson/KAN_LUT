# End-to-End Demo: Kolmogorov-Arnold Network (KAN) LUT Pipeline
# Demonstrates GPU training, static LUT discretization, fixed-point inference,
# and real-time online learning adaptation under a distribution shift.

using KAN_LUT
using Flux
using AMDGPU
using LinearAlgebra
using Random
using JSON

# Set random seed for reproducibility
Random.seed!(42)

# Helper function to generate 2D interleaving half-moons
function make_moons(n_samples=400; noise=0.1)
    n_samples_out = div(n_samples, 2)
    n_samples_in = n_samples - n_samples_out
    
    # Outer moon
    t_out = range(0, pi, length=n_samples_out)
    x_out = cos.(t_out)
    y_out = sin.(t_out)
    
    # Inner moon
    t_in = range(0, pi, length=n_samples_in)
    x_in = 1.0 .- cos.(t_in)
    y_in = 0.5 .- sin.(t_in)
    
    X = hcat(vcat(x_out, x_in), vcat(y_out, y_in))'
    X .+= randn(Float32, size(X)) .* Float32(noise)
    
    # Labels (0 and 1)
    y = vcat(zeros(Int, n_samples_out), ones(Int, n_samples_in))
    Y = Flux.onehotbatch(y, 0:1)
    
    return Float32.(X), Y
end

# Helper to rotate 2D points to simulate a distribution shift
function rotate_points(X::AbstractMatrix{Float32}, theta_degrees::Real)
    rad = Float32(deg2rad(theta_degrees))
    rot_mat = Float32[cos(rad) -sin(rad); sign(rad) cos(rad)]
    # Use exact formula: rot_mat = [cos -sin; sin cos]
    rot_mat = Float32[cos(rad) -sin(rad); sin(rad) cos(rad)]
    return rot_mat * X
end

println("==================================================")
println("STEP 1: Generating Moons Classification Dataset")
println("==================================================")
X_train, Y_train = make_moons(400, noise=0.1f0)
X_test, Y_test = make_moons(100, noise=0.1f0)
println("Train features size: ", size(X_train))
println("Test features size: ", size(X_test))

# Verify domain fits within KAN layer range [-2.5, 2.5]
println("Train input X range: [", minimum(X_train), ", ", maximum(X_train), "]")

println("\n==================================================")
println("STEP 2: Defining and Training KAN on GPU")
println("==================================================")
# Instantiate a 2-layer KAN: 2 inputs -> 8 hidden nodes -> 2 outputs
# Re-create layers with proper domain bounds [-2.5, 2.5]
layer1 = KANLayer(2, 8, G=10, a=-2.5, b=2.5)
layer2 = KANLayer(8, 2, G=10, a=-2.5, b=2.5)
model = Chain(layer1, layer2)

# Select GPU if available
device = AMDGPU.functional() ? roc : cpu
println("Training device: ", device == roc ? "AMD GPU" : "CPU")

# Move model and data to device
model = fmap(device, model)
X_train_dev = device(X_train)
Y_train_dev = device(Y_train)

# Setup optimizer
opt_state = Flux.setup(Flux.AdamW(0.02f0), model)

# Training loop
for epoch in 1:150
    # Forward pass and gradient computation
    loss, grads = Flux.withgradient(model) do m
        logits = m(X_train_dev)
        Flux.Losses.logitcrossentropy(logits, Y_train_dev)
    end
    
    # Update parameters
    Flux.update!(opt_state, model, grads[1])
    
    if epoch % 25 == 0 || epoch == 1
        # Calculate accuracy on training data
        preds = Array(model(X_train_dev))
        acc = sum(argmax(preds, dims=1) .== argmax(Y_train, dims=1)) / size(Y_train, 2)
        println("Epoch $(epoch): Loss = $(round(loss, digits=4)), Accuracy = $(round(acc * 100, digits=2))%")
    end
end

# Move back to CPU for discretization
model_cpu = fmap(cpu, model)
l1 = model_cpu[1]
l2 = model_cpu[2]

# Measure final CPU accuracy
continuous_preds = model_cpu(X_test)
continuous_acc = sum(argmax(continuous_preds, dims=1) .== argmax(Y_test, dims=1)) / size(Y_test, 2)
println("\nFinal Continuous Model Accuracy: $(round(continuous_acc * 100, digits=2))%")

println("\n==================================================")
println("STEP 3: Static LUT Discretization (KANELÉ Style)")
println("==================================================")
# We will discretize both layers to 8-bit inputs, 8-bit outputs, and k=4 fractional bits
n_in = 8
n_out = 8
k = 4

println("Generating L-LUTs for Layer 1...")
luts1 = generate_static_luts(l1, n_in, n_out, k)
println("Generating L-LUTs for Layer 2...")
luts2 = generate_static_luts(l2, n_in, n_out, k)

println("Layer 1 LUT array shape: ", size(luts1))
println("Layer 2 LUT array shape: ", size(luts2))

# Save LUTs to JSON in the same directory as this script
model_path1 = joinpath(@__DIR__, "moons_luts_layer1.json")
model_path2 = joinpath(@__DIR__, "moons_luts_layer2.json")
save_lut_json(model_path1, luts1, l1.a, l1.b, n_in, n_out, k)
save_lut_json(model_path2, luts2, l2.a, l2.b, n_in, n_out, k)
println("Saved Layer 1 LUTs to: ", model_path1)
println("Saved Layer 2 LUTs to: ", model_path2)

println("\n==================================================")
println("STEP 4: Bit-Accurate Fixed-Point Inference")
println("==================================================")
# Quantize the test inputs
X_test_int = quantize_input(X_test, l1.a, l1.b, n_in)

# Run bit-accurate integer LUT inference (adder tree + clipping)
H_int = fixed_lut_inference(luts1, X_test_int, n_out, k)
Y_pred_int = fixed_lut_inference(luts2, H_int, n_out, k)

# Evaluate accuracy
lut_acc = sum(argmax(Y_pred_int, dims=1) .== argmax(Y_test, dims=1)) / size(Y_test, 2)
println("Bit-Accurate Integer LUT Accuracy: $(round(lut_acc * 100, digits=2))%")
println("Accuracy Drop: $(round((continuous_acc - lut_acc) * 100, digits=2))% (typically < 2%)")

println("\n==================================================")
println("STEP 5: Online Learning Adaptation Simulation")
println("==================================================")
# Introduce a distribution shift by rotating the moon dataset by 40 degrees
X_shifted, Y_shifted = make_moons(300, noise=0.1f0)
X_shifted = rotate_points(X_shifted, 40.0f0)

# Continuous model accuracy on shifted data before adaptation
shifted_acc_before = sum(argmax(model_cpu(X_shifted), dims=1) .== argmax(Y_shifted, dims=1)) / size(Y_shifted, 2)
println("Accuracy on shifted dataset (Pre-adaptation): $(round(shifted_acc_before * 100, digits=2))%")

# Simulate on-device online learning (ICML '26 style)
# We will precompute a B-spline basis LUT (degree=1, 8-bit offset)
n_offset = 8
basis_lut, deriv_lut = generate_basis_luts(1, n_offset)

# We will adapt the spline coefficients of KAN Layer 2 using local gradient descent.
# The coefficients are stored in an array of size (d_out, d_in, G + 1)
# Extract the active continuous weights from Layer 2
d_out, d_in = size(l2.w_base)
G = l2.G
coefficients = reshape(l2.w_spline, d_out, d_in, G + 1)

println("Simulating online learning loop for 200 samples...")
# Learning rate for online coefficient updates
lr = 0.05f0
losses = Float32[]

for step in 1:200
    # Pick a random sample from the shifted dataset
    idx = rand(1:size(X_shifted, 2))
    x_val = X_shifted[:, idx:idx] # shape (2, 1)
    y_target = Y_shifted[:, idx:idx] # shape (2, 1)
    
    # 1. Forward Pass: compute hidden activations (using continuous Layer 1 for simplicity)
    h = l1(x_val) # shape (8, 1)
    
    # 2. Local Forward Pass: Online basis LUT inference for Layer 2
    # y = coefficients * BasisLUT(offset)
    y_pred = online_lut_inference(basis_lut, coefficients, h, l2.a, l2.b, G, 1, n_offset) # shape (2, 1)
    
    # 3. Compute Loss & Gradients with respect to outputs (Cross-Entropy Loss)
    probs = softmax(y_pred)
    loss = -sum(y_target .* log.(probs .+ 1f-8))
    push!(losses, loss)
    
    # Output gradient dL/dy = probs - target
    dy = probs .- y_target # shape (2, 1)
    
    # 4. Sparse Weight Update (ICML '26 style):
    # Only update the S+1 active coefficients for each connection.
    # For linear splines, S+1 = 2 active coefficients.
    spacing = (l2.b - l2.a) / G
    for p in 1:d_in
        h_val = clamp(h[p, 1], Float32(l2.a), Float32(l2.b))
        u = (h_val - Float32(l2.a)) / Float32(spacing)
        j = min(floor(Int, u), G - 1)
        offset = u - j
        v_offset = clamp(round(Int, offset * ((1 << n_offset) - 1)), 0, (1 << n_offset) - 1)
        
        # Local Basis values
        b1 = basis_lut[v_offset + 1, 1]
        b2 = basis_lut[v_offset + 1, 2]
        
        # Compute gradient and update active coefficients:
        # coeff_idx = j + 1 and j + 2
        for q in 1:d_out
            grad_c1 = dy[q, 1] * b1
            grad_c2 = dy[q, 1] * b2
            
            coefficients[q, p, j + 1] -= lr * grad_c1
            coefficients[q, p, j + 2] -= lr * grad_c2
        end
    end
    
    if step % 40 == 0 || step == 1
        println("  Step $(step): Online Sample Loss = $(round(loss, digits=4))")
    end
end

# Copy back the updated coefficients to the model
l2_adapted = KANLayer(l2.w_base, reshape(coefficients, d_out, :), l2.a, l2.b, G)
adapted_model = Chain(l1, l2_adapted)

# Calculate final accuracy on shifted dataset post-adaptation
shifted_acc_after = sum(argmax(adapted_model(X_shifted), dims=1) .== argmax(Y_shifted, dims=1)) / size(Y_shifted, 2)
println("Accuracy on shifted dataset (Post-adaptation): $(round(shifted_acc_after * 100, digits=2))%")
println("Adaptation Gain: +$(round((shifted_acc_after - shifted_acc_before) * 100, digits=2))%")

println("\n==================================================")
println("DEMO COMPLETED SUCCESSFULLY!")
println("==================================================")

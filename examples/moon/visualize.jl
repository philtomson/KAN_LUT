# examples/moon/visualize.jl
# Generates a visual 2x2 comparison plot of the decision boundaries:
# 1. Continuous KAN Layer
# 2. Static Bit-Accurate Integer LUT
# 3. Shifted Dataset (Pre-Adaptation)
# 4. Shifted Dataset (Post-Adaptation)

using Pkg
# Ensure Plots package is installed in the project environment
if !haskey(Pkg.project().dependencies, "Plots")
    println("Plots package not found. Installing Plots...")
    Pkg.add("Plots")
end

using KAN_LUT
using Flux
using AMDGPU
using LinearAlgebra
using Random
using Plots

# Set random seed
Random.seed!(42)

# Helper function to generate 2D interleaving half-moons
function make_moons(n_samples=400; noise=0.1)
    n_samples_out = div(n_samples, 2)
    n_samples_in = n_samples - n_samples_out
    
    t_out = range(0, pi, length=n_samples_out)
    x_out = cos.(t_out)
    y_out = sin.(t_out)
    
    t_in = range(0, pi, length=n_samples_in)
    x_in = 1.0 .- cos.(t_in)
    y_in = 0.5 .- sin.(t_in)
    
    X = hcat(vcat(x_out, x_in), vcat(y_out, y_in))'
    X .+= randn(Float32, size(X)) .* Float32(noise)
    
    y = vcat(zeros(Int, n_samples_out), ones(Int, n_samples_in))
    Y = Flux.onehotbatch(y, 0:1)
    
    return Float32.(X), Y
end

# Helper to rotate 2D points to simulate a distribution shift
function rotate_points(X::AbstractMatrix{Float32}, theta_degrees::Real)
    rad = Float32(deg2rad(theta_degrees))
    rot_mat = Float32[cos(rad) -sin(rad); sin(rad) cos(rad)]
    return rot_mat * X
end

println("==================================================")
println("STEP 1: Training KAN model on GPU for Visualization")
println("==================================================")
X_train, Y_train = make_moons(400, noise=0.1f0)
X_test, Y_test = make_moons(100, noise=0.1f0)

# Instantiate KAN Layer Chain
layer1 = KANLayer(2, 8, G=10, a=-2.5, b=2.5)
layer2 = KANLayer(8, 2, G=10, a=-2.5, b=2.5)
model = Chain(layer1, layer2)

# Move to GPU if available
device = AMDGPU.functional() ? roc : cpu
model_dev = fmap(device, model)
X_train_dev = device(X_train)
Y_train_dev = device(Y_train)

# Setup optimizer
opt_state = Flux.setup(Flux.AdamW(0.02f0), model_dev)

# Train model
for epoch in 1:150
    loss, grads = Flux.withgradient(model_dev) do m
        logits = m(X_train_dev)
        Flux.Losses.logitcrossentropy(logits, Y_train_dev)
    end
    Flux.update!(opt_state, model_dev, grads[1])
end

model_cpu = fmap(cpu, model_dev)
l1 = model_cpu[1]
l2 = model_cpu[2]

println("==================================================")
println("STEP 2: Discretizing to Static 8-bit LUTs")
println("==================================================")
n_in = 8
n_out = 8
k = 4
luts1 = generate_static_luts(l1, n_in, n_out, k)
luts2 = generate_static_luts(l2, n_in, n_out, k)

println("==================================================")
println("STEP 3: Simulating Online Learning under Shift")
println("==================================================")
# Rotate moons dataset by 40 degrees
X_shifted, Y_shifted = make_moons(300, noise=0.1f0)
X_shifted = rotate_points(X_shifted, 40.0f0)

# Precompute Basis LUT for online updates
n_offset = 8
basis_lut, _ = generate_basis_luts(1, n_offset)

# Copy spline coefficients for online update
d_out, d_in = size(l2.w_base)
G = l2.G
coefficients = reshape(l2.w_spline, d_out, d_in, G + 1)
lr = 0.05f0

# Online adaptation loop
for step in 1:200
    idx = rand(1:size(X_shifted, 2))
    x_val = X_shifted[:, idx:idx]
    y_target = Y_shifted[:, idx:idx]
    
    h = l1(x_val)
    y_pred = online_lut_inference(basis_lut, coefficients, h, l2.a, l2.b, G, 1, n_offset)
    probs = softmax(y_pred)
    dy = probs .- y_target
    
    spacing = (l2.b - l2.a) / G
    for p in 1:d_in
        h_val = clamp(h[p, 1], Float32(l2.a), Float32(l2.b))
        u = (h_val - Float32(l2.a)) / Float32(spacing)
        j = min(floor(Int, u), G - 1)
        offset = u - j
        v_offset = clamp(round(Int, offset * ((1 << n_offset) - 1)), 0, (1 << n_offset) - 1)
        
        b1 = basis_lut[v_offset + 1, 1]
        b2 = basis_lut[v_offset + 1, 2]
        
        for q in 1:d_out
            coefficients[q, p, j + 1] -= lr * dy[q, 1] * b1
            coefficients[q, p, j + 2] -= lr * dy[q, 1] * b2
        end
    end
end

# Build adapted model
l2_adapted = KANLayer(l2.w_base, reshape(coefficients, d_out, :), l2.a, l2.b, G)
adapted_model = Chain(l1, l2_adapted)

println("==================================================")
println("STEP 4: Generating Plot Image")
println("==================================================")
# Configure GR plotting backend
gr()

x_grid = range(-3.0, 3.0, length=100)
y_grid = range(-3.0, 3.0, length=100)

# Evaluates model predictions over the 2D meshgrid
function get_decision_grid(eval_func)
    grid_vals = zeros(length(x_grid), length(y_grid))
    for (i, xv) in enumerate(x_grid)
        for (j, yv) in enumerate(y_grid)
            inp = Float32[xv; yv]
            out = eval_func(inp)
            # Difference of logits for class 1 vs class 0
            grid_vals[j, i] = out[2] - out[1]
        end
    end
    return grid_vals
end

# 1. Evaluate Continuous KAN decision boundary
grid_continuous = get_decision_grid(x -> model_cpu(x))

# 2. Evaluate Bit-Accurate Integer LUT decision boundary
grid_lut = get_decision_grid(x -> begin
    x_int = quantize_input(reshape(x, :, 1), l1.a, l1.b, n_in)
    h_int = fixed_lut_inference(luts1, x_int, n_out, k)
    y_int = fixed_lut_inference(luts2, h_int, n_out, k)
    y_de = dequantize_output(y_int, l2.a, l2.b, n_out)
    return y_de[:, 1]
end)

# 3. Evaluate Adapted model decision boundary
grid_adapted = get_decision_grid(x -> adapted_model(x))

# Subplot 1: Continuous KAN Boundary
p1 = contourf(x_grid, y_grid, grid_continuous, levels=20, color=:coolwarm, clabel=false, title="Continuous KAN")
scatter!(p1, X_test[1, Y_test[1, :]], X_test[2, Y_test[1, :]], color=:blue, label="Class 0", markersize=3)
scatter!(p1, X_test[1, Y_test[2, :]], X_test[2, Y_test[2, :]], color=:red, label="Class 1", markersize=3)

# Subplot 2: Bit-Accurate LUT Boundary
p2 = contourf(x_grid, y_grid, grid_lut, levels=20, color=:coolwarm, clabel=false, title="Bit-Accurate Integer LUT")
scatter!(p2, X_test[1, Y_test[1, :]], X_test[2, Y_test[1, :]], color=:blue, label=false, markersize=3)
scatter!(p2, X_test[1, Y_test[2, :]], X_test[2, Y_test[2, :]], color=:red, label=false, markersize=3)

# Subplot 3: Shifted Data (Pre-Adaptation)
p3 = contourf(x_grid, y_grid, grid_continuous, levels=20, color=:coolwarm, clabel=false, title="Shifted Data (Pre-Adapt)")
scatter!(p3, X_shifted[1, Y_shifted[1, :]], X_shifted[2, Y_shifted[1, :]], color=:blue, label=false, markersize=3)
scatter!(p3, X_shifted[1, Y_shifted[2, :]], X_shifted[2, Y_shifted[2, :]], color=:red, label=false, markersize=3)

# Subplot 4: Shifted Data (Post-Adaptation)
p4 = contourf(x_grid, y_grid, grid_adapted, levels=20, color=:coolwarm, clabel=false, title="Shifted Data (Post-Adapt)")
scatter!(p4, X_shifted[1, Y_shifted[1, :]], X_shifted[2, Y_shifted[1, :]], color=:blue, label=false, markersize=3)
scatter!(p4, X_shifted[1, Y_shifted[2, :]], X_shifted[2, Y_shifted[2, :]], color=:red, label=false, markersize=3)

# Combine subplots
plt = plot(p1, p2, p3, p4, layout=(2,2), size=(900, 750))

output_path = joinpath(@__DIR__, "moons_plots.png")
savefig(plt, output_path)
println("Successfully saved comparison plot to: ", output_path)

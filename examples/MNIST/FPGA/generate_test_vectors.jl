# examples/MNIST/FPGA/generate_test_vectors.jl
using KAN_LUT
using JSON
using MLDatasets

println("Loading MNIST test data to generate RTL test vectors...")
test_data = MNIST(split=:test)
raw_x = test_data.features[:, :, 1:100] # First 100 samples

# Downsample raw images to 14x14
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

x_down = downsample_mnist(raw_x)
X = reshape(x_down, 196, :)

# Load saved LUTs to compute expected outputs
lut_path1 = joinpath(@__DIR__, "..", "mnist_luts_layer1.json")
lut_path2 = joinpath(@__DIR__, "..", "mnist_luts_layer2.json")

luts1, a1, b1, n_in1, n_out1, k1 = load_lut_json(lut_path1)
luts2, a2, b2, n_in2, n_out2, k2 = load_lut_json(lut_path2)

# Run bit-accurate simulation
X_int = quantize_input(X, a1, b1, n_in1)
H_int = fixed_lut_inference(luts1, X_int, n_out1, k1)
Y_pred_int = fixed_lut_inference(luts2, H_int, n_out2, k2)

# Write to tb_data.txt
out_path = joinpath(@__DIR__, "tb_data.txt")
open(out_path, "w") do io
    for b in 1:100
        # Write inputs (196 bytes in hex)
        for p in 1:196
            val = X_int[p, b]
            write(io, lowercase(string(val, base=16, pad=2)))
        end
        write(io, " ")
        # Write outputs (10 bytes in hex)
        for q in 1:10
            val = Y_pred_int[q, b]
            write(io, lowercase(string(val, base=16, pad=2)))
        end
        write(io, "\n")
    end
end
println("Generated 100 test vectors at: ", out_path)

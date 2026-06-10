# examples/JSC_OpenML/FPGA/generate_test_vectors.jl
using KAN_LUT
using JSON
using NPZ

println("Loading OpenML test data to generate RTL test vectors...")
data_dir = joinpath(@__DIR__, "..", "data")
X_test_raw = npzread(joinpath(data_dir, "X_test.npy"))

# Transpose to (features, samples)
X = Float32.(permutedims(X_test_raw[1:100, :], (2, 1)))

# Load saved LUTs to compute expected outputs
lut_path1 = joinpath(@__DIR__, "..", "jsc_luts_layer1.json")
lut_path2 = joinpath(@__DIR__, "..", "jsc_luts_layer2.json")

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
        # Write inputs (16 bytes in hex)
        for p in 1:16
            val = X_int[p, b]
            write(io, lowercase(string(val, base=16, pad=2)))
        end
        write(io, " ")
        # Write outputs (5 bytes in hex)
        for q in 1:5
            val = Y_pred_int[q, b]
            write(io, lowercase(string(val, base=16, pad=2)))
        end
        write(io, "\n")
    end
end
println("Generated 100 test vectors at: ", out_path)

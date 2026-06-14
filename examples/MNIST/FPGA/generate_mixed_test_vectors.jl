# examples/MNIST/FPGA/generate_mixed_test_vectors.jl
using KAN_LUT
using JSON
using MLDatasets

println("Loading MNIST test data to generate mixed-precision RTL test vectors...")
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

# Helper to load QAT JSON files
function load_qat_lut_json(filename::String)
    lut_dict = open(filename, "r") do io
        JSON.parse(read(io, String))
    end
    meta = lut_dict["metadata"]
    a = Float64(meta["a"])
    b = Float64(meta["b"])
    in_nbits = Int(meta["in_nbits"])
    nbits = Int(meta["nbits"])
    d_in = Int(meta["d_in"])
    d_out = Int(meta["d_out"])
    
    num_entries = 1 << in_nbits
    luts = zeros(Int32, num_entries, d_out, d_in)
    luts_data = lut_dict["luts"]
    for p in 1:d_in
        for q in 1:d_out
            key = "lut_$(q)_$(p)"
            luts[:, q, p] = convert(Vector{Int32}, luts_data[key])
        end
    end
    return luts, a, b, in_nbits, nbits
end

# Load saved mixed-precision LUTs
lut_path1 = joinpath(@__DIR__, "..", "mnist_qat_luts_mixed_layer1.json")
lut_path2 = joinpath(@__DIR__, "..", "mnist_qat_luts_mixed_layer2.json")

luts1, a1, b1, n_in1, n_out1 = load_qat_lut_json(lut_path1)
luts2, a2, b2, n_in2, n_out2 = load_qat_lut_json(lut_path2)

# Binarize inputs to 0 or 1 for simulation (simple thresholding)
x_uint = map(v -> v > 0.3f0 ? Int32(1) : Int32(0), X)

# Run bit-accurate simulation
y_int_l1 = qat_fixed_lut_inference(luts1, x_uint, n_out1, in_nbits=n_in1, is_first_layer=true)
y_int_l2 = qat_fixed_lut_inference(luts2, y_int_l1, n_out2, in_nbits=n_in2, is_first_layer=false)

# Write to tb_data_mixed.txt
out_path = joinpath(@__DIR__, "tb_data_mixed.txt")
open(out_path, "w") do io
    for b in 1:100
        # Write inputs (196 values as 2-digit hex for simple $fscanf reading, e.g. 00 or 01)
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
println("Generated 100 mixed test vectors at: ", out_path)

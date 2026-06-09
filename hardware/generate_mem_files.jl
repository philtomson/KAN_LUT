# hardware/generate_mem_files.jl
# Converts trained JSON LUTs to lane-partitioned hex memory initialization files.

using JSON
using KAN_LUT

function generate_layer_mems(lut_json_path::String, layer_idx::Int, P::Int, data_width::Int, out_dir::String)
    println("Generating mem files for Layer $(layer_idx) (P=$(P), $(data_width)-bit width)...")
    
    # Load JSON
    data = JSON.parsefile(lut_json_path)
    luts_dict = data["luts"]
    metadata = data["metadata"]
    n_in = metadata["n_in"]
    n_out = metadata["n_out"]
    k = metadata["k"]
    
    # Dimensions
    d_out = 0
    d_in = 0
    for key in keys(luts_dict)
        parts = split(key, "_")
        q = parse(Int, parts[2])
        p = parse(Int, parts[3])
        d_out = max(d_out, q)
        d_in = max(d_in, p)
    end
    
    chunks = d_in ÷ P
    lut_depth = 2^n_in
    mem_depth = d_out * chunks * lut_depth
    
    # Pre-parse LUTs into a 3D matrix
    luts_mat = zeros(Int32, lut_depth, d_out, d_in)
    for q in 1:d_out
        for p in 1:d_in
            key = "lut_$(q)_$(p)"
            luts_mat[:, q, p] = luts_dict[key]
        end
    end
    
    # Generate .mem files for each lane
    mask = (1 << data_width) - 1
    
    for l in 0:(P-1)
        out_path = joinpath(out_dir, "layer$(layer_idx)_lane$(l).mem")
        open(out_path, "w") do io
            for q in 0:(d_out-1)
                for c in 0:(chunks-1)
                    p = c * P + l
                    # Fetch LUT for input p and output q
                    lut_vals = luts_mat[:, q + 1, p + 1]
                    for val in lut_vals
                        # Convert to two's complement hex
                        uval = val & mask
                        hex_str = string(uval, base=16)
                        # Pad with leading zeros based on data_width
                        pad_len = ceil(Int, data_width / 4)
                        hex_str = lpad(hex_str, pad_len, '0')
                        write(io, hex_str * "\n")
                    end
                end
            end
        end
        println("  Wrote lane $(l) memory to: $(out_path)")
    end
end

# Main Execution
P = 4
out_dir = @__DIR__

# Paths
layer1_json = joinpath(@__DIR__, "..", "examples", "MNIST", "mnist_luts_layer1.json")
layer2_json = joinpath(@__DIR__, "..", "examples", "MNIST", "mnist_luts_layer2.json")

generate_layer_mems(layer1_json, 1, P, 14, out_dir)
generate_layer_mems(layer2_json, 2, P, 13, out_dir)
println("All memory initialization files generated successfully!")

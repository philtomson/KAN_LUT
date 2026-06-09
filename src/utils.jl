# Utilities for Quantization and JSON Export

using JSON

export quantize_input, dequantize_output, save_lut_json, load_lut_json

"""
    quantize_input(x::AbstractArray{<:Real}, a::Real, b::Real, n_bits::Int)

Quantize a real-valued input matrix/vector to integers in range `0:(2^n_bits - 1)`.
"""
function quantize_input(x::AbstractArray{<:Real}, a::Real, b::Real, n_bits::Int)
    num_levels = (1 << n_bits) - 1
    factor = num_levels / (b - a)
    x_int = clamp.(round.(Int32, (x .- a) .* factor), Int32(0), Int32(num_levels))
    return x_int
end

"""
    dequantize_output(x_int::AbstractArray{<:Integer}, a::Real, b::Real, n_bits::Int)

Dequantize an integer matrix/vector in range `0:(2^n_bits - 1)` back to real numbers.
"""
function dequantize_output(x_int::AbstractArray{<:Integer}, a::Real, b::Real, n_bits::Int)
    num_levels = (1 << n_bits) - 1
    factor = (b - a) / num_levels
    x_real = a .+ Float32.(x_int) .* Float32(factor)
    return x_real
end

"""
    save_lut_json(filename::String, luts::Array{Int32, 3}, a::Real, b::Real, n_in::Int, n_out::Int, k::Int)

Export the L-LUTs and their associated quantization metadata to a JSON file.
"""
function save_lut_json(filename::String, luts::Array{Int32, 3}, a::Real, b::Real, n_in::Int, n_out::Int, k::Int)
    num_entries, d_out, d_in = size(luts)
    
    lut_dict = Dict{String, Any}()
    
    # Store metadata
    lut_dict["metadata"] = Dict(
        "a" => Float64(a),
        "b" => Float64(b),
        "n_in" => Int(n_in),
        "n_out" => Int(n_out),
        "k" => Int(k),
        "d_in" => Int(d_in),
        "d_out" => Int(d_out)
    )
    
    # Store LUT tables for each edge (q, p)
    luts_data = Dict{String, Vector{Int32}}()
    for p in 1:d_in
        for q in 1:d_out
            # JSON keys are 1-indexed for convenience
            key = "lut_$(q)_$(p)"
            luts_data[key] = luts[:, q, p]
        end
    end
    lut_dict["luts"] = luts_data
    
    # Write to file
    open(filename, "w") do io
        JSON.print(io, lut_dict, 2)
    end
end

"""
    load_lut_json(filename::String)

Load L-LUTs and metadata from a JSON file.
Returns a tuple: `(luts, a, b, n_in, n_out, k)`
"""
function load_lut_json(filename::String)
    lut_dict = open(filename, "r") do io
        JSON.parse(read(io, String))
    end
    
    meta = lut_dict["metadata"]
    a = Float64(meta["a"])
    b = Float64(meta["b"])
    n_in = Int(meta["n_in"])
    n_out = Int(meta["n_out"])
    k = Int(meta["k"])
    d_in = Int(meta["d_in"])
    d_out = Int(meta["d_out"])
    
    num_entries = 1 << n_in
    luts = zeros(Int32, num_entries, d_out, d_in)
    
    luts_data = lut_dict["luts"]
    for p in 1:d_in
        for q in 1:d_out
            key = "lut_$(q)_$(p)"
            luts[:, q, p] = convert(Vector{Int32}, luts_data[key])
        end
    end
    
    return luts, a, b, n_in, n_out, k
end

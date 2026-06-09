# KAN LUT Inference Engines

export float_lut_inference, fixed_lut_inference, online_lut_inference

"""
    float_lut_inference(luts::Array{Int32, 3}, x::AbstractMatrix{<:Real}, a::Real, b::Real, n_in::Int, n_out::Int, k::Int)

Reference inference engine using Float conversion.
- `luts`: The Int32 LUT array of shape `(2^n_in, d_out, d_in)`.
- `x`: Real-valued input matrix of shape `(d_in, B)`.
- `a`, `b`: Spline domain boundaries.
- `n_in`, `n_out`: Quantization bitwidths.
- `k`: Fractional precision bits in the LUT.
Returns a real-valued output matrix of shape `(d_out, B)`.
"""
function float_lut_inference(luts::Array{Int32, 3}, x::AbstractMatrix{<:Real}, a::Real, b::Real, n_in::Int, n_out::Int, k::Int)
    d_in = size(luts, 3)
    d_out = size(luts, 2)
    B = size(x, 2)
    
    num_entries = 2^n_in
    δ_in = (b - a) / (num_entries - 1)
    
    # Scale factor used to reconstruct real values from the integer LUT outputs
    # LUT_val = round(y * ((2^n_out - 1) / (b-a)) * 2^k)
    # y = LUT_val / ( ((2^n_out - 1) / (b-a)) * 2^k )
    reconstruct_factor = (b - a) / ((2.0^n_out - 1.0) * (2.0^k))
    
    y = fill(Float32(a), d_out, B)
    
    for b_idx in 1:B
        for p in 1:d_in
            # Clip and map input x[p, b_idx] to quantized index in 0:(num_entries-1)
            x_val = clamp(x[p, b_idx], a, b)
            v = round(Int, (x_val - a) / δ_in)
            
            for q in 1:d_out
                # Accumulate the float representation of LUT outputs
                lut_val = luts[v + 1, q, p]
                y[q, b_idx] += lut_val * reconstruct_factor
            end
        end
    end
    
    return y
end

"""
    fixed_lut_inference(luts::Array{Int32, 3}, x_int::AbstractMatrix{<:Integer}, n_out::Int, k::Int)

Bit-accurate integer inference engine matching the FPGA implementation.
- `luts`: The Int32 LUT array of shape `(2^n_in, d_out, d_in)`.
- `x_int`: Integer-valued input matrix of shape `(d_in, B)`, with values in `0:2^n_in - 1`.
- `n_out`: Output quantization bitwidth.
- `k`: Fractional precision bits in the LUT.
Returns an Int32 matrix of shape `(d_out, B)` containing values in `0:2^n_out - 1`.
"""
function fixed_lut_inference(luts::Array{Int32, 3}, x_int::AbstractMatrix{<:Integer}, n_out::Int, k::Int)
    d_in = size(luts, 3)
    d_out = size(luts, 2)
    B = size(x_int, 2)
    
    y_int = zeros(Int32, d_out, B)
    
    for b_idx in 1:B
        for q in 1:d_out
            sum_val = Int32(0)
            for p in 1:d_in
                # Get the quantized input index
                v = x_int[p, b_idx]
                sum_val += luts[v + 1, q, p]
            end
            
            # Divide sum by 2^k with rounding to nearest integer (hardware-accurate shift)
            y_val = floor(Int32, (sum_val + (1 << (k - 1))) / (1 << k))
            
            # Clip/Saturate to output bitwidth range [0, 2^n_out - 1]
            y_int[q, b_idx] = clamp(y_val, Int32(0), Int32((1 << n_out) - 1))
        end
    end
    
    return y_int
end

"""
    online_lut_inference(basis_lut::Matrix{Float32}, coefficients::Array{Float32, 3}, x::AbstractMatrix{Float32}, a::Real, b::Real, G::Int, degree::Int, n_offset::Int)

Simulate the dynamic forward pass of the online learning mode.
- `basis_lut`: Precomputed basis function values of shape `(2^n_offset, degree + 1)`.
- `coefficients`: Dynamic coefficient array of shape `(d_out, d_in, G + degree)`.
- `x`: Input matrix of shape `(d_in, B)`.
- `a`, `b`: Domain range.
- `G`: Grid size.
- `degree`: Spline degree.
- `n_offset`: Offset bitwidth.
Returns output matrix of shape `(d_out, B)`.
"""
function online_lut_inference(basis_lut::Matrix{Float32}, coefficients::Array{Float32, 3}, x::AbstractMatrix{Float32}, a::Real, b::Real, G::Int, degree::Int, n_offset::Int)
    d_in = size(coefficients, 2)
    d_out = size(coefficients, 1)
    B = size(x, 2)
    
    h = (b - a) / G
    y = zeros(Float32, d_out, B)
    
    for b_idx in 1:B
        for p in 1:d_in
            x_val = clamp(x[p, b_idx], Float32(a), Float32(b))
            
            # Find interval index j in 0:(G-1)
            u = (x_val - Float32(a)) / Float32(h)
            j = min(floor(Int, u), G - 1)
            
            # Find offset within the cell and normalize to [0, 1]
            offset = u - j
            
            # Map normalized offset to quantized index in 0:(2^n_offset - 1)
            v_offset = clamp(round(Int, offset * ((1 << n_offset) - 1)), 0, (1 << n_offset) - 1)
            
            # Look up basis values
            for m in 1:(degree + 1)
                basis_val = basis_lut[v_offset + 1, m]
                
                # Active coefficient index in the dynamic array (1-indexed)
                coeff_idx = j + m
                
                for q in 1:d_out
                    y[q, b_idx] += coefficients[q, p, coeff_idx] * basis_val
                end
            end
        end
    end
    
    return y
end

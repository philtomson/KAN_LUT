# LUT Generation for KAN layers

using NNlib

export evaluate_edge, generate_static_luts, generate_basis_luts

"""
    evaluate_edge(layer::KANLayer, q::Int, p::Int, x::Real)

Evaluate the continuous KAN activation function on edge p -> q at point x.
"""
function evaluate_edge(layer::KANLayer, q::Int, p::Int, x::Real)
    # Base function: w_base * silu(x)
    base_val = layer.w_base[q, p] * silu(x)
    
    # Spline function: sum_i w_spline[q, p_i] * B_i(x)
    h = (layer.b - layer.a) / layer.G
    u = clamp((x - layer.a) / h, 0.0, Float64(layer.G))
    
    spline_val = 0.0
    for i in 0:layer.G
        # linear basis function centered at i
        basis_val = max(0.0, 1.0 - abs(u - i))
        d_in = size(layer.w_base, 2)
        w_idx = p + i * d_in
        spline_val += layer.w_spline[q, w_idx] * basis_val
    end
    
    return base_val + spline_val
end

"""
    generate_static_luts(layer::KANLayer, n_in::Int, n_out::Int, k::Int)

Discretizes a trained continuous `KANLayer` to create integer L-LUTs.
- `n_in`: Number of input quantization bits.
- `n_out`: Number of output quantization bits.
- `k`: Number of fractional bits to preserve in the LUT values.
Returns an Int32 array of shape `(2^n_in, d_out, d_in)` containing scaled integers.
"""
function generate_static_luts(layer::KANLayer, n_in::Int, n_out::Int, k::Int)
    d_in = size(layer.w_base, 2)
    d_out = size(layer.w_base, 1)
    num_entries = 2^n_in
    luts = zeros(Int32, num_entries, d_out, d_in)
    
    δ_in = (layer.b - layer.a) / (num_entries - 1)
    scale_factor = ((2.0^n_out - 1.0) / (layer.b - layer.a)) * (2.0^k)
    
    for p in 1:d_in
        for q in 1:d_out
            for v in 0:(num_entries - 1)
                x = layer.a + v * δ_in
                y = evaluate_edge(layer, q, p, x)
                y_offset = y - layer.a / d_in
                luts[v+1, q, p] = round(Int32, y_offset * scale_factor)
            end
        end
    end
    return luts
end

"""
    generate_basis_luts(degree::Int, n_offset::Int)

Precompute the values of the S+1 active basis functions and their derivatives over the normalized cell interval [0, 1].
- `degree`: Spline degree S (e.g. 1 for linear).
- `n_offset`: Number of quantization bits for the cell offset.
Returns:
- `basis_lut`: Array of shape `(2^n_offset, degree + 1)`
- `deriv_lut`: Array of shape `(2^n_offset, degree + 1)`
"""
function generate_basis_luts(degree::Int, n_offset::Int)
    num_entries = 2^n_offset
    basis_lut = zeros(Float32, num_entries, degree + 1)
    deriv_lut = zeros(Float32, num_entries, degree + 1)
    
    # Setup knot vector for a single interval [0, 1] with boundary padding
    # For degree S, we have knots from -S to S+1
    knots = Float64[]
    for i in -degree:(degree+1)
        push!(knots, i)
    end
    
    for v in 0:(num_entries - 1)
        z = v / (num_entries - 1)  # normalized offset in [0, 1]
        
        # Evaluate basis functions active in [0, 1] (indexes 1 to degree+1 relative to our active set)
        # In the local interval [0, 1], the active basis functions correspond to indices:
        # j = degree + 1, ..., 2*degree + 1 in the knots vector
        for m in 1:(degree + 1)
            j = m
            basis_lut[v+1, m] = Float32(BSpline.cox_de_boor(z, knots, degree, j))
            
            # Derivative: d/dx B_{j,S}(x) = S * [ B_{j,S-1}(x)/(t_{j+S} - t_j) - B_{j+1,S-1}(x)/(t_{j+S+1} - t_{j+1}) ]
            # Here t_{j+S} - t_j = degree, and t_{j+S+1} - t_{j+1} = degree
            if degree > 0
                val1 = BSpline.cox_de_boor(z, knots, degree - 1, j) / degree
                val2 = BSpline.cox_de_boor(z, knots, degree - 1, j + 1) / degree
                deriv_lut[v+1, m] = Float32(degree * (val1 - val2))
            else
                deriv_lut[v+1, m] = 0.0f0
            end
        end
    end
    
    return basis_lut, deriv_lut
end

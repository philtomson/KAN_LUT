# src/qat.jl
# Quantization-Aware Training (QAT) and Dynamic Pruning for KAN_LUT

using Flux
using Functors
using LinearAlgebra
using ChainRulesCore

export QATInputLayer, QATKANLayer, quantize_ste, round_ste, sign_ste
export prune_layer!, get_state_space, generate_qat_luts, qat_fixed_lut_inference

# ==============================================================================
# 1. Straight-Through Estimator (STE) Rounding
# ==============================================================================

"""
    sign_ste(x)

Compute the sign of `x` (-1 or 1). During backward pass, acts as identity (STE).
"""
sign_ste(x::Real) = x < 0.0f0 ? -1.0f0 : 1.0f0
sign_ste(x::AbstractArray) = map(v -> v < 0.0f0 ? -1.0f0 : 1.0f0, x)

function ChainRulesCore.rrule(::typeof(sign_ste), x::AbstractArray)
    y = map(v -> v < 0.0f0 ? -1.0f0 : 1.0f0, x)
    function sign_ste_pullback(Δ)
        return ChainRulesCore.NoTangent(), Δ
    end
    return y, sign_ste_pullback
end

function ChainRulesCore.rrule(::typeof(sign_ste), x::Real)
    y = x < 0.0f0 ? -1.0f0 : 1.0f0
    function sign_ste_pullback(Δ)
        return ChainRulesCore.NoTangent(), Δ
    end
    return y, sign_ste_pullback
end

"""
    round_ste(x)

Round `x` to the nearest integer. During backward pass, acts as identity (STE).
"""
round_ste(x::Real) = round(x)
round_ste(x::AbstractArray) = round.(x)

function ChainRulesCore.rrule(::typeof(round_ste), x::AbstractArray)
    y = round.(x)
    function round_ste_pullback(Δ)
        return ChainRulesCore.NoTangent(), Δ
    end
    return y, round_ste_pullback
end

function ChainRulesCore.rrule(::typeof(round_ste), x::Real)
    y = round(x)
    function round_ste_pullback(Δ)
        return ChainRulesCore.NoTangent(), Δ
    end
    return y, round_ste_pullback
end

# ==============================================================================
# 2. Uniform Symmetric Quantizer Helper
# ==============================================================================

"""
    quantize_ste(x, s::AbstractVector, nbits::Int)

Quantize a continuous tensor `x` to `nbits` signed representation using scale `s`.
Uses Straight-Through Estimator (STE) for rounding.
"""
function quantize_ste(x, s::AbstractVector, nbits::Int)
    if nbits == 1
        return sign_ste(x) .* s
    else
        min_val, max_val = ChainRulesCore.@ignore_derivatives begin
            mn = Float32(-(1 << (nbits - 1)))
            mx = Float32((1 << (nbits - 1)) - 1)
            (mn, mx)
        end
        
        x_scaled = x ./ s
        x_rounded = round_ste(x_scaled)
        x_clamped = clamp.(x_rounded, min_val, max_val)
        return x_clamped .* s
    end
end

# ==============================================================================
# 3. Layer Struct Definitions
# ==============================================================================

"""
    QATInputLayer{B, T, S}

Applies batch normalization, a learnable bias, and quantizes the input features
to `nbits` bits using the learnable `scale` parameter.
"""
struct QATInputLayer{B, T, S}
    bn::B
    bias::T
    scale::S
    nbits::Int
end

Functors.@functor QATInputLayer (bn, bias, scale)

function QATInputLayer(d_in::Int; nbits::Int=8)
    bn = Flux.BatchNorm(d_in)
    # Initialize bias to zeros
    bias = zeros(Float32, d_in)
    # Initialize scale. Assume inputs cover [-1.0, 1.0].
    s_init = nbits == 1 ? 1.0f0 : Float32(1.0 / ((1 << (nbits - 1)) - 1))
    scale = Float32[s_init]
    return QATInputLayer(bn, bias, scale, nbits)
end

function (layer::QATInputLayer)(x::AbstractMatrix)
    # 1. Pre-transforms: BatchNorm and Bias
    x_trans = layer.bn(x) .+ layer.bias
    # 2. Clamp and Quantize to [-1.0, 1.0] range using QuantHardTanh equivalent
    x_clamped = clamp.(x_trans, -1.0f0, 1.0f0)
    return quantize_ste(x_clamped, layer.scale, layer.nbits)
end

"""
    QATKANLayer{W1, W2, T, S, M}

A custom KAN layer that incorporates Quantization-Aware Training (QAT) 
and dynamic pruning masks into the forward pass.
"""
struct QATKANLayer{W1, W2, T, S, M}
    w_base::W1       # Base weight matrix of shape (d_out, d_in)
    w_spline::W2     # Spline coefficients of shape (d_out, d_in * (G + 1))
    a::T             # Spline domain start
    b::T             # Spline domain end
    G::Int           # Grid size
    scale::S         # Learnable scale: 1-element vector
    nbits::Int       # Output quantization bitwidth
    selector::M      # Binary mask matrix of shape (d_out, d_in)
end

Functors.@functor QATKANLayer (w_base, w_spline, scale, selector)

Flux.trainable(layer::QATKANLayer) = (w_base = layer.w_base, w_spline = layer.w_spline, scale = layer.scale)

function QATKANLayer(d_in::Int, d_out::Int; G::Int=10, a::Real=-2.0, b::Real=2.0, nbits::Int=8, init=Flux.glorot_uniform)
    w_base = init(d_out, d_in)
    w_spline = init(d_out, d_in * (G + 1))
    w_spline .= w_spline .* Float32(0.1)
    
    # Initialize learnable scale parameter
    # Since range is [-b, b], init scale is b / (2^(nbits-1) - 1)
    s_init = nbits == 1 ? Float32(abs(b)) : Float32(abs(b) / ((1 << (nbits - 1)) - 1))
    scale = Float32[s_init]
    
    # Pruning selector mask: initially all ones (active)
    selector = ones(Float32, d_out, d_in)
    
    return QATKANLayer(w_base, w_spline, Float32(a), Float32(b), G, scale, nbits, selector)
end

function (layer::QATKANLayer)(x::AbstractMatrix)
    d_in = size(x, 1)
    d_out = size(layer.w_base, 1)
    B = size(x, 2)
    
    # 1. Base activation: shape (d_out, d_in, B)
    base_edge = reshape(layer.w_base, d_out, d_in, 1) .* reshape(silu.(x), 1, d_in, B)
    
    # 2. Spline basis: shape (d_in, G+1, B)
    basis = BSpline.linear_spline_basis(x, layer.a, layer.b, layer.G)
    
    # 3. Spline output: shape (d_out, d_in, B)
    w_spline_reshaped = reshape(layer.w_spline, d_out, d_in, layer.G + 1)
    spline_edge = sum(reshape(w_spline_reshaped, d_out, d_in, layer.G + 1, 1) .* 
                      reshape(basis, 1, d_in, layer.G + 1, B), dims=3)
    spline_edge = dropdims(spline_edge, dims=3)
    
    # 4. Combine base and spline edge outputs
    edge_out = base_edge .+ spline_edge
    
    # 5. Apply pruning mask & Quantize LUT outputs
    edge_out_masked = reshape(layer.selector, d_out, d_in, 1) .* edge_out
    edge_out_q = quantize_ste(edge_out_masked, layer.scale, layer.nbits)
    
    # 6. Sum over input features
    out = sum(edge_out_q, dims=2)
    out = dropdims(out, dims=2) # shape (d_out, B)
    
    # 7. Quantize final node sum
    out_q = quantize_ste(out, layer.scale, layer.nbits)
    
    return out_q
end

# ==============================================================================
# 4. Pruning Logic
# ==============================================================================

"""
    get_state_space(scale::AbstractVector, nbits::Int)

Helper to get the floating-point state space of a quantizer.
"""
function get_state_space(scale::AbstractVector{T}, nbits::Int) where T
    if nbits == 1
        return T[-1.0, 1.0] .* scale[1]
    else
        min_val = -(1 << (nbits - 1))
        max_val = (1 << (nbits - 1)) - 1
        return T.(min_val:max_val) .* scale[1]
    end
end

"""
    prune_layer!(layer::QATKANLayer, threshold::Real, next_layer_selector::Union{Nothing, AbstractMatrix}, input_state_space::AbstractVector)

Perform L2 norm-based connection pruning and propagate backward pruning.
"""
function prune_layer!(layer::QATKANLayer, threshold::Real, next_layer_selector::Union{Nothing, AbstractMatrix}, input_state_space::AbstractVector)
    d_out, d_in = size(layer.w_base)
    
    # Evaluate spline basis functions on input grid
    # input_state_space length: N
    # basis shape: (N, G+1)
    basis = BSpline.linear_spline_basis(input_state_space, layer.a, layer.b, layer.G)
    
    w_spline_reshaped = reshape(layer.w_spline, d_out, d_in, layer.G + 1)
    norms = zeros(Float32, d_out, d_in)
    
    for p in 1:d_in
        for q in 1:d_out
            if layer.selector[q, p] > 0.0f0
                w_qp = w_spline_reshaped[q, p, :]
                # spline_out shape: (N,)
                spline_out = basis * w_qp
                norms[q, p] = norm(spline_out)
            end
        end
    end
    
    # Apply forward pruning threshold
    layer.selector .= layer.selector .* (norms .> threshold)
    
    # Apply backward pruning: if output node q has no active outgoing connections in the next layer,
    # then prune all incoming connections to q in this layer.
    if next_layer_selector !== nothing
        for q in 1:d_out
            # Check if all outgoing connections from node q in next layer are inactive
            if all(next_layer_selector[o, q] == 0.0f0 for o in 1:size(next_layer_selector, 1))
                layer.selector[q, :] .= 0.0f0
            end
        end
    end
    
    return sum(layer.selector) / length(layer.selector)
end

# ==============================================================================
# 5. LUT Discretization & Inference Simulation
# ==============================================================================

"""
    evaluate_edge(layer::QATKANLayer, q::Int, p::Int, x::Real)

Evaluate edge activation q, p on input point x.
"""
function evaluate_edge(layer::QATKANLayer, q::Int, p::Int, x::Real)
    xf = Float32(x)
    base_val = layer.w_base[q, p] * silu(xf)
    
    h = Float32((layer.b - layer.a) / layer.G)
    u = clamp((xf - Float32(layer.a)) / h, 0.0f0, Float32(layer.G))
    
    spline_val = 0.0f0
    for i in 0:layer.G
        basis_val = max(0.0f0, 1.0f0 - abs(u - Float32(i)))
        d_in = size(layer.w_base, 2)
        w_idx = p + i * d_in
        spline_val += layer.w_spline[q, w_idx] * basis_val
    end
    return base_val + spline_val
end

"""
    generate_qat_luts(layer::QATKANLayer, input_scale::Real)

Discretizes a QAT model's KAN layer to signed integer lookup tables.
- `input_scale`: Scale factor of the previous layer's output (or input layer's scale).
Returns an Int32 array of shape `(2^nbits, d_out, d_in)`.
"""
function generate_qat_luts(layer::QATKANLayer, input_scale::Real; in_nbits::Int=layer.nbits)
    d_in = size(layer.w_base, 2)
    d_out = size(layer.w_base, 1)
    num_entries = 1 << in_nbits
    luts = zeros(Int32, num_entries, d_out, d_in)
    
    s_out = layer.scale[1]
    min_val = -(1 << (layer.nbits - 1))
    max_val = (1 << (layer.nbits - 1)) - 1
    
    for p in 1:d_in
        for q in 1:d_out
            if layer.selector[q, p] > 0.0f0
                for v in 0:(num_entries - 1)
                    # Convert index v to signed real input x:
                    x = if in_nbits == 1
                        Float32(2 * v - 1) * Float32(input_scale)
                    else
                        Float32(v - (1 << (in_nbits - 1))) * Float32(input_scale)
                    end
                    y = evaluate_edge(layer, q, p, x)
                    val_int = round(Int32, y / s_out)
                    luts[v+1, q, p] = clamp(val_int, Int32(min_val), Int32(max_val))
                end
            end
        end
    end
    return luts
end

"""
    qat_fixed_lut_inference(luts::Array{Int32, 3}, x_int::AbstractMatrix{<:Integer}, nbits::Int; is_first_layer::Bool=false)

Simulate bit-accurate fixed-point integer inference using adder tree summation and clipping.
"""
function qat_fixed_lut_inference(luts::Array{Int32, 3}, x_int::AbstractMatrix{<:Integer}, out_nbits::Int; in_nbits::Int=out_nbits, is_first_layer::Bool=false)
    d_in = size(luts, 3)
    d_out = size(luts, 2)
    B = size(x_int, 2)
    
    y_int = zeros(Int32, d_out, B)
    min_val = -(1 << (out_nbits - 1))
    max_val = (1 << (out_nbits - 1)) - 1
    
    # Subsequent layers map signed inputs to unsigned indices
    offset = is_first_layer ? 0 : (1 << (in_nbits - 1))
    
    for b_idx in 1:B
        for q in 1:d_out
            sum_val = Int32(0)
            for p in 1:d_in
                # Map input signed/unsigned value to unsigned index
                idx = if is_first_layer
                    x_int[p, b_idx]
                elseif in_nbits == 1
                    div(x_int[p, b_idx] + 1, 2)
                else
                    x_int[p, b_idx] + offset
                end
                sum_val += luts[idx + 1, q, p]
            end
            y_int[q, b_idx] = clamp(sum_val, Int32(min_val), Int32(max_val))
        end
    end
    return y_int
end

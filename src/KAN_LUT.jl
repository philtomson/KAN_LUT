module KAN_LUT

using Flux
using Functors
using LinearAlgebra
using NNlib

# Include submodule files
include("bspline.jl")
using .BSpline

export KANLayer, linear_spline_basis, cox_de_boor, bspline_basis_all

"""
    KANLayer{W1, W2, T}

A custom Flux layer representing a Kolmogorov-Arnold Network (KAN) layer.
Contains learnable base weights and linear spline weights.
Optimized for training on CPU or AMD GPU (via AMDGPU.jl).
"""
struct KANLayer{W1, W2, T}
    w_base::W1       # Base weight matrix of shape (d_out, d_in)
    w_spline::W2     # Spline coefficients of shape (d_out, d_in * (G + 1))
    a::T             # Start of spline domain
    b::T             # End of spline domain
    G::Int           # Grid size
end

# Make KANLayer compatible with Flux (parameters can be moved and trained)
Functors.@functor KANLayer (w_base, w_spline)

"""
    KANLayer(d_in::Int, d_out::Int; G::Int=10, a::Real=-2.0, b::Real=2.0, init=Flux.glorot_uniform)

Outer constructor for KANLayer.
"""
function KANLayer(d_in::Int, d_out::Int; G::Int=10, a::Real=-2.0, b::Real=2.0, init=Flux.glorot_uniform)
    w_base = init(d_out, d_in)
    w_spline = init(d_out, d_in * (G + 1))
    
    # Scale down spline weights slightly to prevent initialization explosion
    w_spline .= w_spline .* Float32(0.1)
    
    return KANLayer(w_base, w_spline, Float32(a), Float32(b), G)
end

# Local silu activation for cross-version compatibility
silu(x) = x * NNlib.sigmoid(x)

function (layer::KANLayer)(x::AbstractMatrix)
    # 1. Base activation output
    base_out = layer.w_base * silu.(x)
    
    # 2. Spline basis evaluation (shape: (d_in, G+1, B))
    basis = linear_spline_basis(x, layer.a, layer.b, layer.G)
    
    # Flatten to (d_in * (G+1), B)
    basis_flat = reshape(basis, :, size(x, 2))
    
    # 3. Spline activation output
    spline_out = layer.w_spline * basis_flat
    
    return base_out .+ spline_out
end

function (layer::KANLayer)(x::AbstractVector)
    x_mat = reshape(x, :, 1)
    out_mat = layer(x_mat)
    return reshape(out_mat, :)
end

# Include other submodules
include("lut_generation.jl")
include("inference.jl")
include("utils.jl")

end # module

module BSpline

using LinearAlgebra
using Adapt

export linear_spline_basis, cox_de_boor, bspline_basis_all

"""
    linear_spline_basis(x::AbstractArray{T}, a::Real, b::Real, G::Int) where T

Compute the degree-1 (linear) B-spline basis function evaluations for input `x`.
The input `x` has shape `(d_in, B)` or is a vector.
Returns a 3D array of shape `(d_in, G+1, B)` where `B` is the batch size.
This implementation is fully vectorized and runs on CPU or AMD GPU (via AMDGPU.jl).
"""
function linear_spline_basis(x::AbstractArray{T}, a::Real, b::Real, G::Int) where T
    h = T((b - a) / G)
    # Clip inputs to the domain [a, b]
    u = clamp.((x .- T(a)) ./ h, T(0.0), T(G))
    
    # Reshape u to (d_in, 1, B) if 2D, or (d_in, 1, 1) if 1D
    nd = ndims(x)
    if nd == 1
        u_reshaped = reshape(u, :, 1, 1)
    else
        u_reshaped = reshape(u, size(u, 1), 1, size(u, 2))
    end
    
    # Create grid indices on the same device as x
    grid = adapt(typeof(x), reshape(T.(0:G), 1, G + 1, 1))
    
    # Compute basis: B_i(x) = max(0, 1 - |u - i|)
    basis = max.(T(0.0), T(1.0) .- abs.(u_reshaped .- grid))
    
    if nd == 1
        return reshape(basis, size(x, 1), G + 1)
    else
        return basis
    end
end

"""
    cox_de_boor(x::Real, knots::Vector{Float64}, degree::Int, j::Int)

Recursive Cox-de Boor formula to evaluate the j-th B-spline basis function of a given degree at x.
"""
function cox_de_boor(x::Real, knots::Vector{Float64}, degree::Int, j::Int)
    if degree == 0
        if knots[j] <= x < knots[j+1]
            return 1.0
        elseif j == length(knots) - 1 && x == knots[end] # handle right boundary
            return 1.0
        else
            return 0.0
        end
    else
        val = 0.0
        denom1 = knots[j+degree] - knots[j]
        if denom1 > 0.0
            val += ((x - knots[j]) / denom1) * cox_de_boor(x, knots, degree - 1, j)
        end
        denom2 = knots[j+degree+1] - knots[j+1]
        if denom2 > 0.0
            val += ((knots[j+degree+1] - x) / denom2) * cox_de_boor(x, knots, degree - 1, j+1)
        end
        return val
    end
end

"""
    bspline_basis_all(x::Real, a::Real, b::Real, G::Int, degree::Int)

Evaluate all B-spline basis functions of a given degree at x.
Returns a vector of length `G + degree`.
"""
function bspline_basis_all(x::Real, a::Real, b::Real, G::Int, degree::Int)
    h = (b - a) / G
    # Define padded knot vector
    knots = Float64[]
    for i in 1:degree
        push!(knots, a)
    end
    for i in 0:G
        push!(knots, a + i * h)
    end
    for i in 1:degree
        push!(knots, b)
    end
    
    num_basis = G + degree
    basis = zeros(Float64, num_basis)
    for j in 1:num_basis
        basis[j] = cox_de_boor(x, knots, degree, j)
    end
    return basis
end

end # module

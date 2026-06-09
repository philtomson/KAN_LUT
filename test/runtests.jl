# Unit Tests for KAN_LUT

using Test
using KAN_LUT
using Flux
using AMDGPU
using LinearAlgebra
using JSON

@testset "KAN_LUT Package Tests" begin

    @testset "B-Spline Basis Properties" begin
        # Linear B-spline basis function partition of unity test:
        # Sum of all basis functions at any point in the domain must equal 1.0.
        a, b, G = -2.0, 2.0, 10
        x = rand(Float32, 2, 5) .* (b - a) .+ a  # random values in [a, b]
        
        basis = linear_spline_basis(x, a, b, G) # shape: (2, G+1, 5)
        
        # Check sum along the basis dimension (dim 2) for each sample and feature
        for b_idx in 1:5
            for f in 1:2
                basis_sum = sum(basis[f, :, b_idx])
                @test basis_sum ≈ 1.0f0 atol=1e-5
            end
        end
    end

    @testset "Continuous KAN Layer Forward/Backward" begin
        d_in, d_out, G = 2, 3, 5
        a, b = -2.0, 2.0
        
        layer = KANLayer(d_in, d_out, G=G, a=a, b=b)
        x = rand(Float32, d_in, 4) .* (b - a) .+ a
        
        # CPU Forward pass
        y_cpu = layer(x)
        @test size(y_cpu) == (d_out, 4)
        
        # CPU Gradient backward pass
        grads = Flux.gradient(l -> sum(l(x)), layer)
        @test grads[1] !== nothing
        @test size(grads[1].w_base) == (d_out, d_in)
        @test size(grads[1].w_spline) == (d_out, d_in * (G + 1))
        
        # GPU compatibility test
        if AMDGPU.functional()
            @info "AMDGPU is functional, running GPU tests..."
            # Move data and model to AMD GPU
            layer_gpu = fmap(roc, layer)
            x_gpu = roc(x)
            
            y_gpu = layer_gpu(x_gpu)
            @test size(y_gpu) == (d_out, 4)
            @test Array(y_gpu) ≈ y_cpu atol=1e-4
            
            # GPU Gradient backward pass
            grads_gpu = Flux.gradient(l -> sum(l(x_gpu)), layer_gpu)
            @test grads_gpu[1] !== nothing
            @test size(grads_gpu[1].w_base) == (d_out, d_in)
            @test size(grads_gpu[1].w_spline) == (d_out, d_in * (G + 1))
        else
            @info "AMDGPU is not active or functional on this system, skipping GPU tests."
        end
    end

    @testset "LUT Generation and File IO" begin
        d_in, d_out, G = 2, 2, 4
        a, b = -2.0, 2.0
        layer = KANLayer(d_in, d_out, G=G, a=a, b=b)
        
        n_in, n_out, k = 8, 8, 4
        luts = generate_static_luts(layer, n_in, n_out, k)
        @test size(luts) == (256, d_out, d_in)
        
        # JSON save and load
        temp_file = tempname() * ".json"
        try
            save_lut_json(temp_file, luts, a, b, n_in, n_out, k)
            luts_loaded, a_l, b_l, n_in_l, n_out_l, k_l = load_lut_json(temp_file)
            
            @test a_l == a
            @test b_l == b
            @test n_in_l == n_in
            @test n_out_l == n_out
            @test k_l == k
            @test luts_loaded == luts
        finally
            rm(temp_file, force=true)
        end
    end

    @testset "LUT Inference Engines" begin
        d_in, d_out, G = 2, 2, 5
        a, b = -2.0, 2.0
        layer = KANLayer(d_in, d_out, G=G, a=a, b=b)
        
        n_in, n_out, k = 8, 8, 4
        luts = generate_static_luts(layer, n_in, n_out, k)
        
        # Test input
        x = Float32[-1.0 1.5; 0.5 -1.8] # size (2, 2)
        
        # Run float LUT inference
        y_float_lut = float_lut_inference(luts, x, a, b, n_in, n_out, k)
        @test size(y_float_lut) == (d_out, 2)
        
        # Run fixed LUT inference
        x_int = quantize_input(x, a, b, n_in)
        y_fixed_lut = fixed_lut_inference(luts, x_int, n_out, k)
        @test size(y_fixed_lut) == (d_out, 2)
        @test all(0 .<= y_fixed_lut .<= 255)
        
        # Check that fixed-point output dequantized matches float-LUT output closely
        y_fixed_dequant = dequantize_output(y_fixed_lut, a, b, n_out)
        @test y_fixed_dequant ≈ y_float_lut atol=0.1
        
        # Check that float-LUT output matches the continuous layer(x) closely
        y_continuous = layer(x)
        @test y_float_lut ≈ y_continuous atol=0.2f0
    end

    @testset "Online Basis LUT Generation" begin
        degree = 1
        n_offset = 8
        basis_lut, deriv_lut = generate_basis_luts(degree, n_offset)
        
        @test size(basis_lut) == (256, 2)
        @test size(deriv_lut) == (256, 2)
        
        # Linear basis partition of unity on cell: B1(z) + B2(z) = 1.0
        for i in 1:256
            @test basis_lut[i, 1] + basis_lut[i, 2] ≈ 1.0f0 atol=1e-5
        end
    end

end

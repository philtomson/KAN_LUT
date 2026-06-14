# examples/MNIST/FPGA/generate_mixed_rtl.jl
using JSON

function get_tree_depth(n::Int)
    return 0
end

function format_signed(nbits::Int, val::Integer)
    if val < 0
        return "-$(nbits)'sd$(abs(val))"
    else
        return "$(nbits)'sd$(val)"
    end
end

function generate_mixed_layer_rtl(lut_json_path::String, layer_idx::Int, sv_out_path::String; is_first_layer::Bool=false)
    println("Generating SystemVerilog for Layer $(layer_idx)...")
    data = JSON.parsefile(lut_json_path)
    metadata = data["metadata"]
    nbits = metadata["nbits"]
    in_nbits = metadata["in_nbits"]
    d_in = metadata["d_in"]
    d_out = metadata["d_out"]
    
    luts_dict = data["luts"]
    
    # 1. Identify active inputs per output q
    active_inputs = Vector{Int}[]
    for q in 1:d_out
        active = Int[]
        for p in 1:d_in
            key = "lut_$(q)_$(p)"
            lut_vals = luts_dict[key]
            if !all(v == 0 for v in lut_vals)
                push!(active, p)
            end
        end
        push!(active_inputs, active)
    end
    
    # 2. Determine layer latency (D_max)
    depths = [get_tree_depth(length(act)) for act in active_inputs]
    D_max = maximum(depths)
    println("Layer $(layer_idx) Max Adder Tree Depth = $(D_max) cycles (Total Latency = $(D_max + 1) cycles)")
    
    open(sv_out_path, "w") do io
        # Write Header
        write(io, """// Auto-generated SystemVerilog for KAN Layer $(layer_idx) (Mixed-Precision & Pruned)
// Input precision: $(in_nbits) bits, Output precision: $(nbits) bits
// Max adder tree depth: $(D_max) cycles

module mnist_kan_layer$(layer_idx) (
    input  logic clk,
    input  logic rst,
    $(is_first_layer ? "input  logic [$(d_in-1):0] in_val," : "input  logic signed [$(d_in-1):0][$(in_nbits-1):0] in_val,")
    output logic signed [$(d_out-1):0][$(nbits-1):0] out_val
);

""")
        
        # We need unique names for signals in the recursive adder tree
        function generate_adder_tree(prefix::String, inputs::Vector{String}, widths::Vector{Int}, stage_idx::Int)
            n = length(inputs)
            if n == 1
                return inputs[1], widths[1]
            end
            
            next_inputs = String[]
            next_widths = Int[]
            num_pairs = div(n, 2)
            has_odd = (n % 2 == 1)
            
            out_width = max(widths...) + 1
            write(io, "  // Neuron $(prefix) - Stage $(stage_idx)\n")
            for i in 0:(num_pairs-1)
                sig_name = "$(prefix)_s$(stage_idx)_$(i)"
                push!(next_inputs, sig_name)
                push!(next_widths, out_width)
                write(io, "  logic signed [$(out_width-1):0] $(sig_name);\n")
            end
            if has_odd
                sig_name = "$(prefix)_s$(stage_idx)_$(num_pairs)"
                push!(next_inputs, sig_name)
                push!(next_widths, widths[end])
                write(io, "  logic signed [$(widths[end]-1):0] $(sig_name);\n")
            end
            write(io, "\n")
            
            write(io, "  always_comb begin\n")
            for i in 0:(num_pairs-1)
                write(io, "    $(next_inputs[i+1]) = $(inputs[2*i+1]) + $(inputs[2*i+2]);\n")
            end
            if has_odd
                write(io, "    $(next_inputs[end]) = $(inputs[end]);\n")
            end
            write(io, "  end\n\n")
            
            return generate_adder_tree(prefix, next_inputs, next_widths, stage_idx + 1)
        end
        
        # Build logic for each output neuron q
        for q in 1:d_out
            active = active_inputs[q]
            n_active = length(active)
            write(io, "  // --- Neuron $(q) (Active inputs: $(n_active)/$(d_in)) ---\n")
            
            if n_active == 0
                # Fully pruned neuron: output is always 0. Generate delay pipeline for 0.
                write(io, "  logic signed [$(nbits-1):0] n$(q)_always_zero [0:$(D_max)];\n")
                write(io, "  always_ff @(posedge clk or posedge rst) begin\n")
                write(io, "    if (rst) begin\n")
                write(io, "      for (int i = 0; i <= $(D_max); i++) n$(q)_always_zero[i] <= '0;\n")
                write(io, "    end else begin\n")
                write(io, "      n$(q)_always_zero[0] <= '0;\n")
                for d in 1:D_max
                    write(io, "      n$(q)_always_zero[$(d)] <= n$(q)_always_zero[$(d-1)];\n")
                end
                write(io, "    end\n")
                write(io, "  end\n")
                write(io, "  assign out_val[$(q-1)] = n$(q)_always_zero[$(D_max)];\n\n")
                continue
            end
            
            # Instantiate combinational ROM lookups only for active inputs
            for p in active
                lut_vals = luts_dict["lut_$(q)_$(p)"]
                write(io, "  logic signed [$(nbits-1):0] n$(q)_rom$(p)_out;\n")
                if is_first_layer
                    # 1-bit input: simple ternary operator
                    val_0 = lut_vals[1]
                    val_1 = lut_vals[2]
                    write(io, "  always_comb n$(q)_rom$(p)_out = in_val[$(p-1)] ? $(format_signed(nbits, val_1)) : $(format_signed(nbits, val_0));\n")
                else
                    # 6-bit signed input: case statement with index shift
                    write(io, "  logic [$(in_nbits-1):0] n$(q)_rom$(p)_idx;\n")
                    write(io, "  assign n$(q)_rom$(p)_idx = in_val[$(p-1)] + $(in_nbits)'sd$(1 << (in_nbits - 1));\n")
                    write(io, "  always_comb begin\n")
                    write(io, "    case (n$(q)_rom$(p)_idx)\n")
                    for (idx, val) in enumerate(lut_vals)
                        write(io, "      $(in_nbits)'d$(idx-1): n$(q)_rom$(p)_out = $(format_signed(nbits, val));\n")
                    end
                    write(io, "      default: n$(q)_rom$(p)_out = $(nbits)'sd0;\n")
                    write(io, "    endcase\n")
                    write(io, "  end\n")
                end
            end
            write(io, "\n")
            
            # Pipelined adder tree for the active ROM outputs
            rom_signals = ["n$(q)_rom$(p)_out" for p in active]
            rom_widths = fill(nbits, n_active)
            final_sum_sig, final_width = generate_adder_tree("n$(q)", rom_signals, rom_widths, 0)
            
            # Delay shift register if tree depth is less than D_max
            tree_depth = depths[q]
            delay_stages = D_max - tree_depth
            
            sat_input_sig = final_sum_sig
            sat_input_width = final_width
            
            if delay_stages > 0
                write(io, "  // Neuron $(q) - Delay pipeline to match Max Depth\n")
                for d in 1:delay_stages
                    write(io, "  logic signed [$(final_width-1):0] n$(q)_delay_$(d);\n")
                end
                write(io, "\n")
                write(io, "  always_ff @(posedge clk or posedge rst) begin\n")
                write(io, "    if (rst) begin\n")
                for d in 1:delay_stages
                    write(io, "      n$(q)_delay_$(d) <= '0;\n")
                end
                write(io, "    end else begin\n")
                write(io, "      n$(q)_delay_1 <= $(final_sum_sig);\n")
                for d in 2:delay_stages
                    write(io, "      n$(q)_delay_$(d) <= n$(q)_delay_$(d-1);\n")
                end
                write(io, "    end\n")
                write(io, "  end\n\n")
                sat_input_sig = "n$(q)_delay_$(delay_stages)"
            end
            
            # Saturation output stage (clamped to [-(1<<(nbits-1)), (1<<(nbits-1))-1])
            min_val = -(1 << (nbits - 1))
            max_val = (1 << (nbits - 1)) - 1
            
            write(io, """  // Neuron $(q) Saturation Stage
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      out_val[$(q-1)] <= '0;
    end else begin
      out_val[$(q-1)] <= ($(sat_input_sig) > $(max_val)) ? $(nbits)'sd$(max_val) :
                         ($(sat_input_sig) < $(min_val)) ? -$(nbits)'sd$(abs(min_val)) :
                         $(sat_input_sig)[$(nbits-1):0];
    end
  end

""")
        end
        write(io, "endmodule\n")
    end
    println("Finished Layer $(layer_idx) -> $(sv_out_path)")
    return D_max + 1
end

function generate_top_rtl(sv_out_path::String, l1_latency::Int, l2_latency::Int)
    println("Generating SystemVerilog Top wrapper...")
    open(sv_out_path, "w") do io
        write(io, """// Auto-generated SystemVerilog Top wrapper for Mixed-Precision MNIST KAN
// Cascades Layer 1 and Layer 2.
// Total core latency: $(l1_latency + l2_latency) clock cycles

module mnist_kan_top (
    input  logic clk,
    input  logic rst,
    input  logic [195:0] in_val,
    output logic signed [9:0][5:0] out_val
);

  // Intermediate Layer 1 outputs
  logic signed [63:0][5:0] l1_out;

  // Layer 1 Instance (latency = $(l1_latency) cycles)
  mnist_kan_layer1 l1 (
      .clk(clk),
      .rst(rst),
      .in_val(in_val),
      .out_val(l1_out)
  );

  // Layer 2 Instance (latency = $(l2_latency) cycles)
  mnist_kan_layer2 l2 (
      .clk(clk),
      .rst(rst),
      .in_val(l1_out),
      .out_val(out_val)
  );

endmodule
""")
    end
    println("Finished Top wrapper -> $(sv_out_path)")
end

# Main Execution
layer1_json = joinpath(@__DIR__, "..", "mnist_qat_luts_mixed_layer1.json")
layer2_json = joinpath(@__DIR__, "..", "mnist_qat_luts_mixed_layer2.json")

layer1_sv = joinpath(@__DIR__, "mnist_kan_layer1_mixed.sv")
layer2_sv = joinpath(@__DIR__, "mnist_kan_layer2_mixed.sv")
top_sv = joinpath(@__DIR__, "mnist_kan_top_mixed.sv")

l1_lat = generate_mixed_layer_rtl(layer1_json, 1, layer1_sv, is_first_layer=true)
l2_lat = generate_mixed_layer_rtl(layer2_json, 2, layer2_sv, is_first_layer=false)
generate_top_rtl(top_sv, l1_lat, l2_lat)

println("\nAll SystemVerilog files generated successfully!")
println("Layer 1 Latency: $(l1_lat) cycles")
println("Layer 2 Latency: $(l2_lat) cycles")
println("Total core latency: $(l1_lat + l2_lat) clock cycles")

# examples/MNIST/FPGA/generate_rtl.jl
using JSON

function generate_layer_rtl(lut_json_path::String, layer_idx::Int, sv_out_path::String)
    println("Generating SystemVerilog for Layer $(layer_idx)...")
    
    # Load JSON
    data = JSON.parsefile(lut_json_path)
    # Extract metadata
    metadata = data["metadata"]
    a = Float32(metadata["a"])
    b = Float32(metadata["b"])
    n_in = metadata["n_in"]
    n_out = metadata["n_out"]
    k = metadata["k"]
    
    # Extract LUT dictionary
    luts_dict = data["luts"]
    
    # Find global min/max across all LUT values to compute optimal bitwidth
    all_vals = Int[]
    for v in values(luts_dict)
        append!(all_vals, v)
    end
    min_val = minimum(all_vals)
    max_val = maximum(all_vals)
    
    function bits_needed(min_v, max_v)
        N = 1
        while true
            min_bound = -(1 << (N-1))
            max_bound = (1 << (N-1)) - 1
            if min_v >= min_bound && max_v <= max_bound
                return N
            end
            N += 1
        end
    end
    rom_width = bits_needed(min_val, max_val)
    println("Optimal bitwidth for Layer $(layer_idx) ROM outputs: $(rom_width) bits (Range: [$(min_val), $(max_val)])")

    # Dimensions: we parse the keys to find d_in and d_out
    # keys are like "lut_q_p" (where q is 1:d_out, p is 1:d_in)
    d_out = 0
    d_in = 0
    for key in keys(luts_dict)
        parts = split(key, "_")
        q = parse(Int, parts[2])
        p = parse(Int, parts[3])
        d_out = max(d_out, q)
        d_in = max(d_in, p)
    end
    
    println("Layer $(layer_idx) dimensions: $(d_in) -> $(d_out)")
    
    open(sv_out_path, "w") do io
        # Write Header
        write(io, """// Auto-generated SystemVerilog for KAN Layer $(layer_idx)
// Target Domain: [$(a), $(b)]
// Quantization: n_in=$(n_in), n_out=$(n_out), k=$(k)

module mnist_kan_layer$(layer_idx) (
    input  logic clk,
    input  logic rst,
    input  logic [$(n_in-1):0] in_val [0:$(d_in-1)],
    output logic [$(n_out-1):0] out_val [0:$(d_out-1)]
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
            
            write(io, "  always_ff @(posedge clk or posedge rst) begin\n")
            write(io, "    if (rst) begin\n")
            for sig in next_inputs
                write(io, "      $(sig) <= '0;\n")
            end
            write(io, "    end else begin\n")
            for i in 0:(num_pairs-1)
                write(io, "      $(next_inputs[i+1]) <= $(inputs[2*i+1]) + $(inputs[2*i+2]);\n")
            end
            if has_odd
                write(io, "      $(next_inputs[end]) <= $(inputs[end]);\n")
            end
            write(io, "    end\n")
            write(io, "  end\n\n")
            
            return generate_adder_tree(prefix, next_inputs, next_widths, stage_idx + 1)
        end

        # For each output neuron, we first instantiate combinational ROMs for all inputs
        for q in 1:d_out
            write(io, "  // --- Neuron $(q) LUT ROMs ---\n")
            for p in 1:d_in
                lut_vals = luts_dict["lut_$(q)_$(p)"]
                write(io, "  logic signed [$(rom_width-1):0] n$(q)_rom$(p)_out;\n")
                write(io, "  always_comb begin\n")
                write(io, "    case (in_val[$(p-1)])\n")
                for (idx, val) in enumerate(lut_vals)
                    hex_str = string(idx - 1, base = 16)
                    if length(hex_str) == 1
                        hex_str = "0" * hex_str
                    end
                    if val < 0
                        write(io, "      8'h$(hex_str): n$(q)_rom$(p)_out = -$(rom_width)'sd$(abs(val));\n")
                    else
                        write(io, "      8'h$(hex_str): n$(q)_rom$(p)_out = $(rom_width)'sd$(val);\n")
                    end
                end
                write(io, "      default: n$(q)_rom$(p)_out = $(rom_width)'sd0;\n")
                write(io, "    endcase\n")
                write(io, "  end\n")
            end
            write(io, "\n")
            
            # Now build the pipelined adder tree for this neuron
            rom_signals = ["n$(q)_rom$(p)_out" for p in 1:d_in]
            rom_widths = fill(rom_width, d_in)
            final_sum_sig, final_width = generate_adder_tree("n$(q)", rom_signals, rom_widths, 0)
            
            # Divide and Saturate
            write(io, """  // Neuron $(q) Division and Saturation
  logic signed [$(final_width-1):0] n$(q)_div_val;
  assign n$(q)_div_val = ( $(final_sum_sig) + $(final_width)'sd$(2^(k-1)) ) >>> $(k);

  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      out_val[$(q-1)] <= '0;
    end else begin
      out_val[$(q-1)] <= (n$(q)_div_val > 255) ? 8'hFF : (n$(q)_div_val < 0) ? 8'h00 : n$(q)_div_val[7:0];
    end
  end

""")
        end
        
        # End Module
        write(io, "endmodule\n")
    end
    println("Finished Layer $(layer_idx) -> $(sv_out_path)")
end

function generate_top_rtl(sv_out_path::String, d_out_l1::Int)
    println("Generating SystemVerilog Top wrapper with Layer 1 output size $(d_out_l1)...")
    open(sv_out_path, "w") do io
        write(io, """// Auto-generated SystemVerilog Top wrapper for MNIST KAN
// Cascades Layer 1 and Layer 2 with an inter-layer register boundary.

module mnist_kan_top (
    input  logic clk,
    input  logic rst,
    input  logic [7:0] in_val [0:195],
    output logic [7:0] out_val [0:9]
);

  // Intermediate Layer 1 outputs
  logic [7:0] l1_out [0:$(d_out_l1-1)];

  // Layer 1 Instance
  mnist_kan_layer1 l1 (
      .clk(clk),
      .rst(rst),
      .in_val(in_val),
      .out_val(l1_out)
  );

  // Layer 2 Instance
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
layer1_json = joinpath(@__DIR__, "..", "mnist_luts_layer1.json")
layer2_json = joinpath(@__DIR__, "..", "mnist_luts_layer2.json")

layer1_sv = joinpath(@__DIR__, "mnist_kan_layer1.sv")
layer2_sv = joinpath(@__DIR__, "mnist_kan_layer2.sv")
top_sv = joinpath(@__DIR__, "mnist_kan_top.sv")

generate_layer_rtl(layer1_json, 1, layer1_sv)
generate_layer_rtl(layer2_json, 2, layer2_sv)

# Get Layer 1 output dimension dynamically
l1_data = JSON.parsefile(layer1_json)
l1_luts = l1_data["luts"]
d_out_l1 = 0
for key in keys(l1_luts)
    parts = split(key, "_")
    global d_out_l1 = max(d_out_l1, parse(Int, parts[2]))
end

generate_top_rtl(top_sv, d_out_l1)
println("All SystemVerilog files generated successfully!")

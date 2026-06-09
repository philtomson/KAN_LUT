// hardware/mnist_generic_top.sv
// Top-level MNIST generic KAN-LUT accelerator.
// Integrates Layer 1 and Layer 2 KAN cores and memory buffers.

`timescale 1ns/1ps

module mnist_generic_top #(
    parameter L1_INIT_0 = "",
    parameter L1_INIT_1 = "",
    parameter L1_INIT_2 = "",
    parameter L1_INIT_3 = "",
    parameter L2_INIT_0 = "",
    parameter L2_INIT_1 = "",
    parameter L2_INIT_2 = "",
    parameter L2_INIT_3 = ""
)(
    input  logic       clk,
    input  logic       rst,
    
    // Control Interface
    input  logic       start,
    output logic       done,
    
    // Testbench Write Port to Input Activation Memory
    input  logic       in_mem_we,
    input  logic [7:0] in_mem_addr,
    input  logic [7:0] in_mem_din,
    
    // Testbench Read Port from Output Activation Memory
    input  logic [3:0] out_mem_addr,
    output logic [7:0] out_mem_dout
);

  // Activation Memories
  logic [7:0] in_mem [0:195];
  logic [7:0] layer1_out_mem [0:31];
  logic [7:0] output_mem [0:9];

  // Write inputs from testbench
  always_ff @(posedge clk) begin
    if (in_mem_we) begin
      in_mem[in_mem_addr] <= in_mem_din;
    end
  end

  // Read outputs to testbench
  assign out_mem_dout = output_mem[out_mem_addr];

  // Control FSM State
  typedef enum logic [1:0] {
      IDLE,
      RUN_L1,
      RUN_L2,
      DONE_STATE
  } state_t;
  
  state_t state;

  // Layer 1 Signals
  logic        l1_start;
  logic        l1_done;
  logic [5:0]  l1_in_bram_addr; // 196 / 4 = 49 chunks -> 6 bits
  logic [31:0] l1_in_bram_dout;
  logic [4:0]  l1_out_bram_addr; // 32 outputs -> 5 bits
  logic [7:0]  l1_out_bram_din;
  logic        l1_out_bram_we;

  // Map 1D in_mem to 32-bit chunk output for Layer 1
  assign l1_in_bram_dout = {
    in_mem[{l1_in_bram_addr, 2'b11}],
    in_mem[{l1_in_bram_addr, 2'b10}],
    in_mem[{l1_in_bram_addr, 2'b01}],
    in_mem[{l1_in_bram_addr, 2'b00}]
  };

  // Write Layer 1 outputs to layer1_out_mem
  always_ff @(posedge clk) begin
    if (l1_out_bram_we) begin
      layer1_out_mem[l1_out_bram_addr] <= l1_out_bram_din;
    end
  end

  // Layer 2 Signals
  logic        l2_start;
  logic        l2_done;
  logic [2:0]  l2_in_bram_addr; // 32 / 4 = 8 chunks -> 3 bits
  logic [31:0] l2_in_bram_dout;
  logic [3:0]  l2_out_bram_addr; // 10 outputs -> 4 bits
  logic [7:0]  l2_out_bram_din;
  logic        l2_out_bram_we;

  // Map 1D layer1_out_mem to 32-bit chunk output for Layer 2
  assign l2_in_bram_dout = {
    layer1_out_mem[{l2_in_bram_addr, 2'b11}],
    layer1_out_mem[{l2_in_bram_addr, 2'b10}],
    layer1_out_mem[{l2_in_bram_addr, 2'b01}],
    layer1_out_mem[{l2_in_bram_addr, 2'b00}]
  };

  // Write Layer 2 outputs to output_mem
  always_ff @(posedge clk) begin
    if (l2_out_bram_we) begin
      output_mem[l2_out_bram_addr] <= l2_out_bram_din;
    end
  end

  // Instantiate Layer 1 Generic Core (196 -> 32)
  // LUTs use 14-bit data width, fractional bits = 4
  kan_generic_core #(
      .INPUT_DIM(196),
      .OUTPUT_DIM(32),
      .PARALLELISM(4),
      .LUT_DEPTH(256),
      .DATA_WIDTH(14),
      .FRACTIONAL_BITS(4),
      .INIT_FILE_0(L1_INIT_0),
      .INIT_FILE_1(L1_INIT_1),
      .INIT_FILE_2(L1_INIT_2),
      .INIT_FILE_3(L1_INIT_3)
  ) l1_core (
      .clk(clk),
      .rst(rst),
      .start(l1_start),
      .done(l1_done),
      .in_bram_addr(l1_in_bram_addr),
      .in_bram_dout(l1_in_bram_dout),
      .out_bram_addr(l1_out_bram_addr),
      .out_bram_din(l1_out_bram_din),
      .out_bram_we(l1_out_bram_we),
      
      // Loader ports unused in simulation
      .loader_we(1'b0),
      .loader_lane('0),
      .loader_addr('0),
      .loader_din('0)
  );

  // Instantiate Layer 2 Generic Core (32 -> 10)
  // LUTs use 13-bit data width, fractional bits = 4
  kan_generic_core #(
      .INPUT_DIM(32),
      .OUTPUT_DIM(10),
      .PARALLELISM(4),
      .LUT_DEPTH(256),
      .DATA_WIDTH(13),
      .FRACTIONAL_BITS(4),
      .INIT_FILE_0(L2_INIT_0),
      .INIT_FILE_1(L2_INIT_1),
      .INIT_FILE_2(L2_INIT_2),
      .INIT_FILE_3(L2_INIT_3)
  ) l2_core (
      .clk(clk),
      .rst(rst),
      .start(l2_start),
      .done(l2_done),
      .in_bram_addr(l2_in_bram_addr),
      .in_bram_dout(l2_in_bram_dout),
      .out_bram_addr(l2_out_bram_addr),
      .out_bram_din(l2_out_bram_din),
      .out_bram_we(l2_out_bram_we),
      
      // Loader ports unused in simulation
      .loader_we(1'b0),
      .loader_lane('0),
      .loader_addr('0),
      .loader_din('0)
  );

  // Orchestrator FSM
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      state    <= IDLE;
      l1_start <= 1'b0;
      l2_start <= 1'b0;
      done     <= 1'b0;
    end else begin
      l1_start <= 1'b0;
      l2_start <= 1'b0;
      done     <= 1'b0;
      
      case (state)
        IDLE: begin
          if (start) begin
            l1_start <= 1'b1;
            state    <= RUN_L1;
          end
        end
        
        RUN_L1: begin
          if (l1_done) begin
            l2_start <= 1'b1;
            state    <= RUN_L2;
          end
        end
        
        RUN_L2: begin
          if (l2_done) begin
            done  <= 1'b1;
            state <= IDLE;
          end
        end
        
        default: state <= IDLE;
      endcase
    end
  end

endmodule

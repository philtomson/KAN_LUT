// examples/JSC_OpenML/FPGA/jsc_generic_top.sv
// Top-level JSC_OpenML generic KAN-LUT accelerator.
// Integrates Layer 1 and Layer 2 KAN cores with a unified PSRAM interface.

`timescale 1ns/1ps

module jsc_generic_top (
    input  logic       clk,
    input  logic       rst,
    
    // Control Interface
    input  logic       start,
    output logic       done,
    
    // Testbench Write Port to Input Activation Memory (16 inputs)
    input  logic       in_mem_we,
    input  logic [3:0] in_mem_addr,
    input  logic [7:0] in_mem_din,
    
    // Testbench Read Port from Output Activation Memory (5 outputs)
    input  logic [2:0] out_mem_addr,
    output logic [7:0] out_mem_dout,
    
    // External PSRAM Interface
    output logic        mem_req,
    output logic [31:0] mem_addr,
    input  logic [15:0] mem_rdata,
    input  logic        mem_rvalid,
    input  logic        mem_ready
);

  // Activation Memories
  logic [7:0] in_mem [0:15];
  logic [7:0] layer1_out_mem [0:7];
  logic [7:0] output_mem [0:4];

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

  // Layer 1 Signals (16 -> 8)
  logic        l1_start;
  logic        l1_done;
  logic [1:0]  l1_in_bram_addr; // 16 / 4 = 4 chunks -> 2 bits
  logic [31:0] l1_in_bram_dout;
  logic [2:0]  l1_out_bram_addr; // 8 outputs -> 3 bits
  logic [7:0]  l1_out_bram_din;
  logic        l1_out_bram_we;
  
  logic        l1_mem_req;
  logic [31:0] l1_mem_addr;
  logic [15:0] l1_mem_rdata;
  logic        l1_mem_rvalid;
  logic        l1_mem_ready;

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

  // Layer 2 Signals (8 -> 5)
  logic        l2_start;
  logic        l2_done;
  logic [0:0]  l2_in_bram_addr; // 8 / 4 = 2 chunks -> 1 bit
  logic [31:0] l2_in_bram_dout;
  logic [2:0]  l2_out_bram_addr; // 5 outputs -> 3 bits
  logic [7:0]  l2_out_bram_din;
  logic        l2_out_bram_we;
  
  logic        l2_mem_req;
  logic [31:0] l2_mem_addr;
  logic [15:0] l2_mem_rdata;
  logic        l2_mem_rvalid;
  logic        l2_mem_ready;

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

  // Memory Interface Multiplexer (Arbiter)
  always_comb begin
    if (state == RUN_L1 || l1_start) begin
      mem_req       = l1_mem_req;
      mem_addr      = l1_mem_addr;
      l1_mem_rdata  = mem_rdata;
      l1_mem_rvalid = mem_rvalid;
      l1_mem_ready  = mem_ready;
      
      l2_mem_rdata  = '0;
      l2_mem_rvalid = 1'b0;
      l2_mem_ready  = 1'b0;
    end else begin
      mem_req       = l2_mem_req;
      // Offset Layer 2 weights so they reside contiguously after Layer 1 (offset = 16 * 8 * 256 = 32,768 words)
      mem_addr      = l2_mem_addr + 32'd32768;
      l2_mem_rdata  = mem_rdata;
      l2_mem_rvalid = mem_rvalid;
      l2_mem_ready  = mem_ready;
      
      l1_mem_rdata  = '0;
      l1_mem_rvalid = 1'b0;
      l1_mem_ready  = 1'b0;
    end
  end

  // Instantiate Layer 1 Generic Core (16 -> 8)
  kan_generic_core #(
      .INPUT_DIM(16),
      .OUTPUT_DIM(8),
      .PARALLELISM(4),
      .LUT_DEPTH(256),
      .DATA_WIDTH(16),
      .FRACTIONAL_BITS(4)
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
      .mem_req(l1_mem_req),
      .mem_addr(l1_mem_addr),
      .mem_rdata(l1_mem_rdata),
      .mem_rvalid(l1_mem_rvalid),
      .mem_ready(l1_mem_ready)
  );

  // Instantiate Layer 2 Generic Core (8 -> 5)
  kan_generic_core #(
      .INPUT_DIM(8),
      .OUTPUT_DIM(5),
      .PARALLELISM(4),
      .LUT_DEPTH(256),
      .DATA_WIDTH(13),
      .FRACTIONAL_BITS(4)
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
      .mem_req(l2_mem_req),
      .mem_addr(l2_mem_addr),
      .mem_rdata(l2_mem_rdata),
      .mem_rvalid(l2_mem_rvalid),
      .mem_ready(l2_mem_ready)
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

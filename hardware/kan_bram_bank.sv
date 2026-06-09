// hardware/kan_bram_bank.sv
// Parameterized Block RAM bank for KAN LUT entries.
// Supports synchronous read and write operations.

`timescale 1ns/1ps

module kan_bram_bank #(
    parameter int DATA_WIDTH = 12,
    parameter int ADDR_WIDTH = 19,
    parameter INIT_FILE = ""
)(
    input  logic                  clk,
    
    // Read Port A
    input  logic [ADDR_WIDTH-1:0] addr_a,
    output logic signed [DATA_WIDTH-1:0] dout_a,
    
    // Write Port B (for bootloading weights)
    input  logic                  we_b,
    input  logic [ADDR_WIDTH-1:0] addr_b,
    input  logic signed [DATA_WIDTH-1:0] din_b,
    output logic signed [DATA_WIDTH-1:0] dout_b
);

  // Declare memory array
  localparam int MEM_DEPTH = 1 << ADDR_WIDTH;
  logic signed [DATA_WIDTH-1:0] mem [0:MEM_DEPTH-1];

  // Optional memory initialization from file
  initial begin
    if (INIT_FILE != "") begin
      $readmemh(INIT_FILE, mem);
    end
  end

  // Port A - Read Only
  always_ff @(posedge clk) begin
    dout_a <= mem[addr_a];
  end

  // Port B - Read/Write
  always_ff @(posedge clk) begin
    if (we_b) begin
      mem[addr_b] <= din_b;
    end
    dout_b <= mem[addr_b];
  end

endmodule

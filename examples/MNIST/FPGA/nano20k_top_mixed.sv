// examples/MNIST/FPGA/nano20k_top_mixed.sv
// Top-level FPGA wrapper for Mixed-Precision Unrolled KAN on Sipeed Tang Nano 20K.
// Receives 196 bytes of image data over UART at 2,000,000 baud, runs inference in 2 cycles,
// and returns the 1-byte predicted digit class (0-9).

`timescale 1ns/1ps

module nano20k_top_mixed (
    input  wire sys_clk,   // 27 MHz
    input  wire s1,        // Button (active-low reset helper)
    input  wire uart_rx,
    output wire uart_tx,
    
    // Status LEDs
    output wire [5:0] led
);

  wire clk;       // 54 MHz
  wire clk_sdram; // unused but required by PLL primitive

  Gowin_rPLL pll(
      .clkout(clk),
      .clkoutp(clk_sdram),
      .clkin(sys_clk)
  );

  // Reset generation
  reg [15:0] rst_cnt = 0;
  reg rst = 1;
  always @(posedge clk) begin
      if (rst_cnt != 16'hFFFF) begin
          rst_cnt <= rst_cnt + 1;
          rst <= 1;
      end else begin
          rst <= 0;
      end
  end

  // UART RX & TX Instantiation
  wire [7:0] rx_data;
  wire rx_valid;
  uart_rx #(.CLK_FREQ(54_000_000), .BAUD_RATE(2_000_000)) u_rx (
      .clk(clk), .rst(rst), .rx(uart_rx), .data(rx_data), .valid(rx_valid)
  );

  reg tx_start;
  reg [7:0] tx_data;
  wire tx_busy;
  uart_tx #(.CLK_FREQ(54_000_000), .BAUD_RATE(2_000_000)) u_tx (
      .clk(clk), .rst(rst), .start(tx_start), .data(tx_data), .tx(uart_tx), .busy(tx_busy)
  );

  // 196-bit input buffer for Layer 1
  reg [195:0] kan_in;
  reg [7:0] byte_cnt;

  // Mixed-Precision KAN Core Instance
  wire signed [9:0][5:0] kan_out;
  
  mnist_kan_top u_core (
      .clk(clk),
      .rst(rst),
      .in_val(kan_in),
      .out_val(kan_out)
  );

  // Control FSM States
  localparam S_RX_WAIT  = 0;
  localparam S_RUN_CORE = 1;
  localparam S_ARGMAX   = 2;
  localparam S_TX_RES   = 3;

  reg [1:0] state = S_RX_WAIT;
  reg [1:0] cycle_cnt;
  reg signed [5:0] max_score;
  reg [3:0] best_class;
  reg [3:0] d_idx;

  // Active-low status LEDs (lit LEDs are 0)
  assign led = ~{4'b0000, state};

  always @(posedge clk or posedge rst) begin
      if (rst) begin
          state <= S_RX_WAIT;
          byte_cnt <= 0;
          kan_in <= 0;
          tx_start <= 0;
          tx_data <= 0;
          cycle_cnt <= 0;
          max_score <= 6'b100000; // -32 (minimum possible 6-bit signed)
          best_class <= 0;
          d_idx <= 0;
      end else begin
          tx_start <= 0;
          case (state)
              S_RX_WAIT: begin
                  if (rx_valid) begin
                      // Store received byte into the 196-bit input buffer.
                      // The drawing app sends 196 bytes (0..255).
                      // Threshold to 1-bit input: 1 if > 0, else 0.
                      kan_in[byte_cnt] <= (rx_data > 8'd0) ? 1'b1 : 1'b0;
                      
                      byte_cnt <= byte_cnt + 1;
                      if (byte_cnt == 8'd195) begin
                          state <= S_RUN_CORE;
                          cycle_cnt <= 0;
                      end
                  end
              end

              S_RUN_CORE: begin
                  // Wait 2 clock cycles for combinational propagation through registers
                  cycle_cnt <= cycle_cnt + 1;
                  if (cycle_cnt == 2'd2) begin
                      state <= S_ARGMAX;
                      d_idx <= 0;
                      max_score <= 6'b100000;
                      best_class <= 0;
                  end
              end

              S_ARGMAX: begin
                  if (d_idx < 4'd10) begin
                      if (kan_out[d_idx] > max_score || d_idx == 0) begin
                          max_score <= kan_out[d_idx];
                          best_class <= d_idx;
                      end
                      d_idx <= d_idx + 1;
                  end else begin
                      state <= S_TX_RES;
                  end
              end

              S_TX_RES: begin
                  if (!tx_busy && !tx_start) begin
                      tx_start <= 1;
                      tx_data <= {4'd0, best_class};
                      byte_cnt <= 0;
                      state <= S_RX_WAIT;
                  end
              end
          endcase
      end
  end

endmodule

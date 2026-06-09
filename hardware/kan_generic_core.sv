// hardware/kan_generic_core.sv
// Parameterized KAN-LUT Generic IP Core with a 3-stage pipeline.

`timescale 1ns/1ps

module kan_generic_core #(
    parameter int INPUT_DIM = 196,
    parameter int OUTPUT_DIM = 32,
    parameter int PARALLELISM = 4,
    parameter int LUT_DEPTH = 256,
    parameter int DATA_WIDTH = 14,
    parameter int FRACTIONAL_BITS = 4,
    parameter INIT_FILE_0 = "",
    parameter INIT_FILE_1 = "",
    parameter INIT_FILE_2 = "",
    parameter INIT_FILE_3 = ""
)(
    input  logic clk,
    input  logic rst,
    
    // Control Interface
    input  logic start,
    output logic done,
    
    // Input Activation BRAM Interface (width is PARALLELISM * 8 bits)
    output logic [$clog2(INPUT_DIM/PARALLELISM)-1:0] in_bram_addr,
    input  logic [(PARALLELISM*8)-1:0]              in_bram_dout,
    
    // Output Activation BRAM Interface
    output logic [$clog2(OUTPUT_DIM)-1:0]           out_bram_addr,
    output logic [7:0]                              out_bram_din,
    output logic                                    out_bram_we,
    
    // Weight Loader Port (for loading LUT BRAMs at boot time)
    input  logic                                    loader_we,
    input  logic [$clog2(PARALLELISM)-1:0]           loader_lane,
    input  logic [$clog2(OUTPUT_DIM * (INPUT_DIM/PARALLELISM) * LUT_DEPTH)-1:0] loader_addr,
    input  logic signed [DATA_WIDTH-1:0]            loader_din
);

  localparam int CHUNKS = INPUT_DIM / PARALLELISM;
  localparam int BRAM_ADDR_WIDTH = $clog2(OUTPUT_DIM * CHUNKS * LUT_DEPTH);

  // BRAM bank interface signals
  logic [BRAM_ADDR_WIDTH-1:0]       bram_addr_a [0:PARALLELISM-1];
  logic signed [DATA_WIDTH-1:0]     bram_dout_a [0:PARALLELISM-1];
  
  // Explicitly instantiate 4 lane BRAMs for maximum tool compatibility
  kan_bram_bank #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(BRAM_ADDR_WIDTH),
      .INIT_FILE(INIT_FILE_0)
  ) lane_ram_0 (
      .clk(clk),
      .addr_a(bram_addr_a[0]),
      .dout_a(bram_dout_a[0]),
      .we_b(loader_we && (loader_lane == 0)),
      .addr_b(loader_addr),
      .din_b(loader_din),
      .dout_b()
  );

  kan_bram_bank #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(BRAM_ADDR_WIDTH),
      .INIT_FILE(INIT_FILE_1)
  ) lane_ram_1 (
      .clk(clk),
      .addr_a(bram_addr_a[1]),
      .dout_a(bram_dout_a[1]),
      .we_b(loader_we && (loader_lane == 1)),
      .addr_b(loader_addr),
      .din_b(loader_din),
      .dout_b()
  );

  kan_bram_bank #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(BRAM_ADDR_WIDTH),
      .INIT_FILE(INIT_FILE_2)
  ) lane_ram_2 (
      .clk(clk),
      .addr_a(bram_addr_a[2]),
      .dout_a(bram_dout_a[2]),
      .we_b(loader_we && (loader_lane == 2)),
      .addr_b(loader_addr),
      .din_b(loader_din),
      .dout_b()
  );

  kan_bram_bank #(
      .DATA_WIDTH(DATA_WIDTH),
      .ADDR_WIDTH(BRAM_ADDR_WIDTH),
      .INIT_FILE(INIT_FILE_3)
  ) lane_ram_3 (
      .clk(clk),
      .addr_a(bram_addr_a[3]),
      .dout_a(bram_dout_a[3]),
      .we_b(loader_we && (loader_lane == 3)),
      .addr_b(loader_addr),
      .din_b(loader_din),
      .dout_b()
  );

  // Pipeline Signals
  typedef enum logic [1:0] {
      IDLE,
      RUN,
      WRITE_OUT,
      DONE_STATE
  } state_t;
  
  state_t state;
  
  logic [$clog2(OUTPUT_DIM)-1:0] q_reg;
  logic [$clog2(CHUNKS)-1:0]     c_s0;
  
  // Pipeline Stage 1 Registers
  logic [$clog2(CHUNKS)-1:0]     c_s1;
  logic [$clog2(OUTPUT_DIM)-1:0] q_s1;
  logic                          val_en_s1;
  logic [(PARALLELISM*8)-1:0]    in_bram_dout_reg;
  
  // Pipeline Stage 2 Registers
  logic [$clog2(CHUNKS)-1:0]     c_s2;
  logic [$clog2(OUTPUT_DIM)-1:0] q_s2;
  logic                          val_en_s2;
  
  localparam int ACC_WIDTH = DATA_WIDTH + $clog2(INPUT_DIM);
  logic signed [ACC_WIDTH-1:0]   acc;
  logic signed [ACC_WIDTH-1:0]   div_val;

  // Adder tree for Stage 2 BRAM outputs
  // Combinational summation of all lane outputs
  logic signed [ACC_WIDTH-1:0] temp_sum;
  logic signed [DATA_WIDTH+$clog2(PARALLELISM)-1:0] lane_sum;
  always_comb begin
    temp_sum = 0;
    for (int i = 0; i < PARALLELISM; i = i + 1) begin
      temp_sum = temp_sum + bram_dout_a[i];
    end
    lane_sum = temp_sum[DATA_WIDTH+$clog2(PARALLELISM)-1:0];
  end

  // Stage 0: Address the Input BRAM
  assign in_bram_addr = c_s0;

  // Stage 1: Route input values to BRAM banks
  // For each lane l, address the BRAM using the input value from in_bram_dout
  always_comb begin
    for (int i = 0; i < PARALLELISM; i = i + 1) begin
      // Calculate BRAM address: (q_s1 * CHUNKS + c_s1) * 256 + x_val
      bram_addr_a[i] = (BRAM_ADDR_WIDTH) ' ( ( (q_s1 * CHUNKS) + c_s1 ) * LUT_DEPTH + in_bram_dout_reg[i*8 +: 8] );
    end
  end

  // Control and Pipeline Sequence
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      state         <= IDLE;
      q_reg         <= '0;
      c_s0          <= '0;
      c_s1          <= '0;
      q_s1          <= '0;
      val_en_s1     <= 1'b0;
      in_bram_dout_reg <= '0;
      c_s2          <= '0;
      q_s2          <= '0;
      val_en_s2     <= 1'b0;
      acc           <= '0;
      out_bram_we   <= 1'b0;
      out_bram_addr <= '0;
      out_bram_din  <= '0;
      done          <= 1'b0;
    end else begin
      out_bram_we      <= 1'b0;
      done             <= 1'b0;
      in_bram_dout_reg <= in_bram_dout;
      
      case (state)
        IDLE: begin
          if (start) begin
            state     <= RUN;
            q_reg     <= '0;
            c_s0      <= '0;
            val_en_s1 <= 1'b0;
            val_en_s2 <= 1'b0;
            acc       <= '0;
          end
        end
        
        RUN: begin
          // Stage 0 -> Stage 1 Pipeline Register Shift
          c_s1      <= c_s0;
          q_s1      <= q_reg;
          val_en_s1 <= 1'b1;
          
          // Stage 1 -> Stage 2 Pipeline Register Shift
          c_s2      <= c_s1;
          q_s2      <= q_s1;
          val_en_s2 <= val_en_s1;
          
          // Accumulate Stage 2 inputs
          if (val_en_s2) begin
            acc <= acc + lane_sum;
          end
          
          // Stage 0 Counter Logic
          if (c_s0 == (CHUNKS - 1)) begin
            // We have addressed all inputs for the current neuron q_reg.
            // Move to drainage state or transition to write output.
            state <= WRITE_OUT;
            c_s0  <= '0;
          end else begin
            c_s0  <= c_s0 + 1;
          end
        end
        
        WRITE_OUT: begin
          // Drain the remaining pipeline stages
          c_s1      <= '0;
          val_en_s1 <= 1'b0;
          
          c_s2      <= c_s1;
          q_s2      <= q_s1;
          val_en_s2 <= val_en_s1;
          
          if (val_en_s2) begin
            acc <= acc + lane_sum;
          end
          
          // Wait until Stage 2 has drained the last chunk of the current neuron
          if (val_en_s2 && (c_s2 == (CHUNKS - 1))) begin
            // Perform division and saturation
            // Divide by 2^FRACTIONAL_BITS (rounding shift)
            div_val = (acc + lane_sum + (1 << (FRACTIONAL_BITS - 1))) >>> FRACTIONAL_BITS;
            
            // Saturation/Clipping to [0, 255]
            out_bram_we   <= 1'b1;
            out_bram_addr <= q_s2;
            out_bram_din  <= (div_val > 255) ? 8'hFF : (div_val < 0) ? 8'h00 : div_val[7:0];
            
            // Next neuron sequence
            if (q_reg == (OUTPUT_DIM - 1)) begin
              state <= DONE_STATE;
            end else begin
              q_reg <= q_reg + 1;
              state <= RUN;
              c_s0  <= '0;
              acc   <= '0;
            end
          end
        end
        
        DONE_STATE: begin
          done  <= 1'b1;
          state <= IDLE;
        end
        
        default: state <= IDLE;
      endcase
    end
  end

endmodule

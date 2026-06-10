// hardware/kan_generic_core.sv
// Parameterized KAN-LUT Generic IP Core with PSRAM-backed Weight Cache.

`timescale 1ns/1ps

module kan_generic_core #(
    parameter int INPUT_DIM = 196,
    parameter int OUTPUT_DIM = 32,
    parameter int PARALLELISM = 4,
    parameter int LUT_DEPTH = 256,
    parameter int DATA_WIDTH = 14,
    parameter int FRACTIONAL_BITS = 4
)(
    input  logic clk,
    input  logic rst,
    
    // Control Interface
    input  logic start,
    output logic done,
    
    // Input Activation BRAM Interface
    output logic [$clog2(INPUT_DIM/PARALLELISM)-1:0] in_bram_addr,
    input  logic [(PARALLELISM*8)-1:0]              in_bram_dout,
    
    // Output Activation BRAM Interface
    output logic [$clog2(OUTPUT_DIM)-1:0]           out_bram_addr,
    output logic [7:0]                              out_bram_din,
    output logic                                    out_bram_we,
    
    // External PSRAM Interface
    output logic                                    mem_req,
    output logic [31:0]                             mem_addr,
    input  logic [15:0]                             mem_rdata,
    input  logic                                    mem_rvalid,
    input  logic                                    mem_ready
);

  localparam int CHUNKS = INPUT_DIM / PARALLELISM;

  // Local weight caches (4 lanes)
  // Stores CHUNKS weights per lane for the active neuron
  logic signed [DATA_WIDTH-1:0] lane_cache [0:PARALLELISM-1][0:CHUNKS-1];
  
  // Pipeline Signals
  typedef enum logic [2:0] {
      IDLE,
      PREFETCH,
      RUN,
      WRITE_OUT,
      DONE_STATE
  } state_t;
  
  state_t state;
  
  typedef enum logic [1:0] {
      MEM_CMD,
      MEM_WAIT
  } mem_state_t;
  mem_state_t mem_state;
  
  logic [15:0] p_fetch;
  
  logic [$clog2(OUTPUT_DIM)-1:0] q_reg;
  logic [$clog2(CHUNKS)-1:0]     c_s0;
  
  // Pipeline Stage 1 Registers
  logic [$clog2(CHUNKS)-1:0]     c_s1;
  logic [$clog2(OUTPUT_DIM)-1:0] q_s1;
  logic                          val_en_s1;
  
  // Pipeline Stage 2 Registers
  logic [$clog2(CHUNKS)-1:0]     c_s2;
  logic [$clog2(OUTPUT_DIM)-1:0] q_s2;
  logic                          val_en_s2;
  logic signed [DATA_WIDTH-1:0]  bram_dout_a [0:PARALLELISM-1];
  
  localparam int ACC_WIDTH = DATA_WIDTH + $clog2(INPUT_DIM);
  logic signed [ACC_WIDTH-1:0]   acc;
  logic signed [ACC_WIDTH-1:0]   div_val;

  // Adder tree for Stage 2 BRAM outputs
  logic signed [ACC_WIDTH-1:0] temp_sum;
  logic signed [DATA_WIDTH+$clog2(PARALLELISM)-1:0] lane_sum;
  always_comb begin
    temp_sum = 0;
    for (int i = 0; i < PARALLELISM; i = i + 1) begin
      temp_sum = temp_sum + bram_dout_a[i];
    end
    lane_sum = temp_sum[DATA_WIDTH+$clog2(PARALLELISM)-1:0];
  end

  // Input Activation BRAM address multiplexing
  always_comb begin
    if (state == PREFETCH) begin
      in_bram_addr = (p_fetch >> $clog2(PARALLELISM));
    end else if (state == RUN) begin
      in_bram_addr = c_s0;
    end else begin
      in_bram_addr = '0;
    end
  end

  // Synchronous read of lane_cache to match the 1-cycle latency of the original BRAM bank
  always_ff @(posedge clk) begin
    if (state == RUN || state == WRITE_OUT) begin
      for (int i = 0; i < PARALLELISM; i = i + 1) begin
        bram_dout_a[i] <= lane_cache[i][c_s1];
      end
    end
  end

  // Control and Pipeline Sequence
  always_ff @(posedge clk or posedge rst) begin
    if (rst) begin
      state         <= IDLE;
      mem_state     <= MEM_CMD;
      p_fetch       <= '0;
      q_reg         <= '0;
      c_s0          <= '0;
      c_s1          <= '0;
      q_s1          <= '0;
      val_en_s1     <= 1'b0;
      c_s2          <= '0;
      q_s2          <= '0;
      val_en_s2     <= 1'b0;
      acc           <= '0;
      out_bram_we   <= 1'b0;
      out_bram_addr <= '0;
      out_bram_din  <= '0;
      mem_req       <= 1'b0;
      mem_addr      <= '0;
      done          <= 1'b0;
    end else begin
      out_bram_we   <= 1'b0;
      done          <= 1'b0;
      mem_req       <= 1'b0;
      
      case (state)
        IDLE: begin
          if (start) begin
            state     <= PREFETCH;
            q_reg     <= '0;
            p_fetch   <= '0;
            mem_state <= MEM_CMD;
          end
        end
        
        PREFETCH: begin
          case (mem_state)
            MEM_CMD: begin
              mem_req  <= 1'b1;
              // Compute memory address: (q_reg * INPUT_DIM + p_fetch) * 256 + x_val
              mem_addr <= ( (q_reg * INPUT_DIM + p_fetch) * 256 ) + in_bram_dout[(p_fetch[1:0])*8 +: 8];
              if (mem_ready) begin
                mem_state <= MEM_WAIT;
              end
            end
            
            MEM_WAIT: begin
              if (mem_rvalid) begin
                lane_cache[p_fetch[1:0]][p_fetch >> 2] <= mem_rdata[DATA_WIDTH-1:0];
                if (p_fetch == (INPUT_DIM - 1)) begin
                  state     <= RUN;
                  c_s0      <= '0;
                  val_en_s1 <= 1'b0;
                  val_en_s2 <= 1'b0;
                  acc       <= '0;
                end else begin
                  p_fetch   <= p_fetch + 1;
                  mem_state <= MEM_CMD;
                end
              end
            end
            
            default: mem_state <= MEM_CMD;
          endcase
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
            div_val = (acc + lane_sum + (1 << (FRACTIONAL_BITS - 1))) >>> FRACTIONAL_BITS;
            
            // Saturation/Clipping to [0, 255]
            out_bram_we   <= 1'b1;
            out_bram_addr <= q_s2;
            out_bram_din  <= (div_val > 255) ? 8'hFF : (div_val < 0) ? 8'h00 : div_val[7:0];
            
            // Next neuron sequence
            if (q_reg == (OUTPUT_DIM - 1)) begin
              state <= DONE_STATE;
            end else begin
              q_reg     <= q_reg + 1;
              state     <= PREFETCH;
              p_fetch   <= '0;
              mem_state <= MEM_CMD;
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

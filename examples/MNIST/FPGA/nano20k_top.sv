`timescale 1ns/1ps

module nano20k_top (
    input  wire sys_clk,   // 27 MHz
    input  wire s1,        // Button
    input  wire uart_rx,
    output wire uart_tx,

    // Tang Nano 20k Embedded SDRAM Magic Ports
    output wire O_sdram_clk,
    output wire O_sdram_cke,
    output wire O_sdram_cs_n,
    output wire O_sdram_cas_n,
    output wire O_sdram_ras_n,
    output wire O_sdram_wen_n,
    inout  wire [31:0] IO_sdram_dq,
    output wire [10:0] O_sdram_addr,
    output wire [1:0] O_sdram_ba,
    output wire [3:0] O_sdram_dqm,
    
    // Status LEDs
    output wire [5:0] led
);

wire clk;       // 54 MHz
wire clk_sdram; // 54 MHz (phase shifted)

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

// SDRAM Controller
wire sdram_busy;
reg sdram_rd, sdram_wr, sdram_refresh;
reg [22:0] sdram_addr;
reg [7:0] sdram_din;
wire [7:0] sdram_dout;
wire [31:0] sdram_dout32;
wire sdram_data_ready;

sdram #(.FREQ(54_000_000)) u_sdram (
    .clk(clk), .clk_sdram(clk_sdram), .resetn(~rst),
    .addr(sdram_addr), .rd(sdram_rd), .wr(sdram_wr), .refresh(sdram_refresh),
    .din(sdram_din), .dout(sdram_dout), .dout32(sdram_dout32),
    .data_ready(sdram_data_ready), .busy(sdram_busy),
    
    .SDRAM_DQ(IO_sdram_dq),
    .SDRAM_A(O_sdram_addr),
    .SDRAM_BA(O_sdram_ba),
    .SDRAM_nCS(O_sdram_cs_n),
    .SDRAM_nWE(O_sdram_wen_n),
    .SDRAM_nRAS(O_sdram_ras_n),
    .SDRAM_nCAS(O_sdram_cas_n),
    .SDRAM_CLK(O_sdram_clk),
    .SDRAM_CKE(O_sdram_cke),
    .SDRAM_DQM(O_sdram_dqm)
);

// Auto-Refresh Generator (15 us)
localparam REFRESH_COUNT = 54_000_000 / 1000 / 1000 * 15;
reg [11:0] refresh_time = 0;
reg refresh_needed = 0;
always @(posedge clk) begin
    if (rst) begin
        refresh_time <= 0;
        refresh_needed <= 0;
    end else begin
        if (refresh_time < REFRESH_COUNT) begin
            refresh_time <= refresh_time + 1;
        end else begin
            refresh_needed <= 1;
        end
        if (sdram_refresh) begin
            refresh_time <= 0;
            refresh_needed <= 0;
        end
    end
end

// UART RX & TX
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

// KAN Core
reg kan_start;
wire kan_done;
reg in_mem_we;
reg [7:0] in_mem_addr;
reg [7:0] in_mem_din;

reg [3:0] out_mem_addr;
wire [7:0] out_mem_dout;

wire kan_mem_req;
wire [31:0] kan_mem_addr;
reg [15:0] kan_mem_rdata;
reg kan_mem_rvalid;
wire kan_mem_ready = !sdram_busy && !refresh_needed;

mnist_generic_top u_core (
    .clk(clk),
    .rst(rst),
    .start(kan_start),
    .done(kan_done),
    .in_mem_we(in_mem_we),
    .in_mem_addr(in_mem_addr),
    .in_mem_din(in_mem_din),
    .out_mem_addr(out_mem_addr),
    .out_mem_dout(out_mem_dout),
    .mem_req(kan_mem_req),
    .mem_addr(kan_mem_addr),
    .mem_rdata(kan_mem_rdata),
    .mem_rvalid(kan_mem_rvalid),
    .mem_ready(kan_mem_ready)
);

// State Machine
localparam S_BOOT_WAIT   = 0;
localparam S_LOAD_WEIGHT = 1;
localparam S_INFER_WAIT  = 2;
localparam S_INFER_RUN   = 3;
localparam S_ARGMAX      = 4;
localparam S_TX_RESULT   = 5;

reg [3:0] state = S_BOOT_WAIT;
reg [22:0] weight_bytes_loaded = 0;
localparam TOTAL_WEIGHT_BYTES = 6750208; // (3,211,264 + 163,840) words * 2 bytes

reg [7:0] infer_bytes_loaded = 0;
reg [7:0] max_score;
reg [7:0] best_class;

// Memory arbiter state
reg mem_reading = 0;
reg [31:0] kan_read_addr = 0;

assign led = ~{1'b0, refresh_needed, state};

always @(posedge clk) begin
    if (rst) begin
        state <= S_BOOT_WAIT;
        weight_bytes_loaded <= 0;
        infer_bytes_loaded <= 0;
        sdram_wr <= 0;
        sdram_rd <= 0;
        sdram_refresh <= 0;
        kan_start <= 0;
        tx_start <= 0;
        in_mem_we <= 0;
        kan_mem_rvalid <= 0;
        mem_reading <= 0;
    end else begin
        sdram_wr <= 0;
        sdram_rd <= 0;
        sdram_refresh <= 0;
        kan_start <= 0;
        tx_start <= 0;
        in_mem_we <= 0;
        kan_mem_rvalid <= 0;

        // Arbitration between FSM and KAN Core
        if (refresh_needed && !sdram_busy && !mem_reading && state != S_LOAD_WEIGHT) begin
            // Highest priority is refresh (except when hammering writes in boot)
            sdram_refresh <= 1;
        end else if (mem_reading) begin
            if (sdram_data_ready) begin
                mem_reading <= 0;
                kan_mem_rvalid <= 1;
                // Select proper 16-bit word from 32-bit row
                if (kan_read_addr[1] == 1'b0) begin
                    kan_mem_rdata <= sdram_dout32[15:0];
                end else begin
                    kan_mem_rdata <= sdram_dout32[31:16];
                end
            end
        end else if (state == S_INFER_RUN && kan_mem_req && !sdram_busy && !refresh_needed) begin
            // KAN core read request
            mem_reading <= 1;
            sdram_rd <= 1;
            kan_read_addr <= kan_mem_addr << 1; // Convert 16-bit word addr to byte addr
            sdram_addr <= kan_mem_addr << 1;
        end else begin
            // Main FSM
            case (state)
                S_BOOT_WAIT: begin
                    if (rx_valid) begin
                        state <= S_LOAD_WEIGHT;
                        if (!sdram_busy) begin
                            sdram_wr <= 1;
                            sdram_addr <= weight_bytes_loaded;
                            sdram_din <= rx_data;
                            weight_bytes_loaded <= weight_bytes_loaded + 1;
                        end
                    end
                end
                S_LOAD_WEIGHT: begin
                    // Handle refresh during long boot load
                    if (refresh_needed && !sdram_busy) begin
                        sdram_refresh <= 1;
                    end else if (rx_valid) begin
                        if (!sdram_busy) begin
                            sdram_wr <= 1;
                            sdram_addr <= weight_bytes_loaded;
                            sdram_din <= rx_data;
                            weight_bytes_loaded <= weight_bytes_loaded + 1;
                            if (weight_bytes_loaded + 1 == TOTAL_WEIGHT_BYTES) begin
                                state <= S_INFER_WAIT;
                            end
                        end
                    end
                end
                S_INFER_WAIT: begin
                    if (rx_valid) begin
                        in_mem_we <= 1;
                        in_mem_addr <= infer_bytes_loaded;
                        in_mem_din <= rx_data;
                        infer_bytes_loaded <= infer_bytes_loaded + 1;
                        if (infer_bytes_loaded == 195) begin
                            state <= S_INFER_RUN;
                            infer_bytes_loaded <= 0;
                            kan_start <= 1;
                        end
                    end
                end
                S_INFER_RUN: begin
                    if (kan_done) begin
                        state <= S_ARGMAX;
                        out_mem_addr <= 0;
                        max_score <= 0;
                        best_class <= 0;
                    end
                end
                S_ARGMAX: begin
                    if (out_mem_addr < 10) begin
                        if (out_mem_dout > max_score || out_mem_addr == 0) begin
                            max_score <= out_mem_dout;
                            best_class <= {4'd0, out_mem_addr};
                        end
                        out_mem_addr <= out_mem_addr + 1;
                    end else begin
                        state <= S_TX_RESULT;
                    end
                end
                S_TX_RESULT: begin
                    if (!tx_busy && !tx_start) begin
                        tx_start <= 1;
                        tx_data <= best_class;
                        state <= S_INFER_WAIT; // Loop back
                    end
                end
            endcase
        end
    end
end

endmodule

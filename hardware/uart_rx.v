module uart_rx #(
    parameter CLK_FREQ = 54_000_000,
    parameter BAUD_RATE = 2_000_000
)(
    input wire clk,
    input wire rst,
    input wire rx,
    output reg [7:0] data,
    output reg valid
);

localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;
localparam HALF_CLKS = CLKS_PER_BIT / 2;

reg [3:0] state;
reg [15:0] clk_cnt;
reg [2:0] bit_cnt;
reg [7:0] shift_reg;

localparam IDLE = 0, START = 1, DATA = 2, STOP = 3;

// Double flop RX for metastability
reg rx_d1, rx_d2;
always @(posedge clk) begin
    rx_d1 <= rx;
    rx_d2 <= rx_d1;
end

always @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        valid <= 0;
        clk_cnt <= 0;
        bit_cnt <= 0;
    end else begin
        valid <= 0;
        case (state)
            IDLE: begin
                if (rx_d2 == 0) begin
                    state <= START;
                    clk_cnt <= 1;
                end
            end
            START: begin
                if (clk_cnt == HALF_CLKS) begin
                    if (rx_d2 == 0) begin
                        clk_cnt <= 1;
                        state <= DATA;
                        bit_cnt <= 0;
                    end else begin
                        state <= IDLE;
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
            DATA: begin
                if (clk_cnt == CLKS_PER_BIT) begin
                    clk_cnt <= 1;
                    shift_reg <= {rx_d2, shift_reg[7:1]};
                    if (bit_cnt == 7) begin
                        state <= STOP;
                    end else begin
                        bit_cnt <= bit_cnt + 1;
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
            STOP: begin
                if (clk_cnt == CLKS_PER_BIT) begin
                    clk_cnt <= 0;
                    state <= IDLE;
                    if (rx_d2 == 1) begin
                        data <= shift_reg;
                        valid <= 1;
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
        endcase
    end
end
endmodule

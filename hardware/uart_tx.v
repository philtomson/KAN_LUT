module uart_tx #(
    parameter CLK_FREQ = 54_000_000,
    parameter BAUD_RATE = 2_000_000
)(
    input wire clk,
    input wire rst,
    input wire start,
    input wire [7:0] data,
    output reg tx,
    output reg busy
);

localparam CLKS_PER_BIT = CLK_FREQ / BAUD_RATE;

reg [3:0] state;
reg [15:0] clk_cnt;
reg [2:0] bit_cnt;
reg [7:0] shift_reg;

localparam IDLE = 0, START = 1, DATA = 2, STOP = 3;

always @(posedge clk) begin
    if (rst) begin
        state <= IDLE;
        tx <= 1;
        busy <= 0;
        clk_cnt <= 0;
        bit_cnt <= 0;
    end else begin
        case (state)
            IDLE: begin
                tx <= 1;
                if (start) begin
                    state <= START;
                    shift_reg <= data;
                    busy <= 1;
                    clk_cnt <= 1;
                    tx <= 0; // Start bit
                end else begin
                    busy <= 0;
                end
            end
            START: begin
                if (clk_cnt == CLKS_PER_BIT) begin
                    clk_cnt <= 1;
                    state <= DATA;
                    tx <= shift_reg[0];
                    shift_reg <= {1'b0, shift_reg[7:1]};
                    bit_cnt <= 0;
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
            DATA: begin
                if (clk_cnt == CLKS_PER_BIT) begin
                    clk_cnt <= 1;
                    if (bit_cnt == 7) begin
                        state <= STOP;
                        tx <= 1; // Stop bit
                    end else begin
                        bit_cnt <= bit_cnt + 1;
                        tx <= shift_reg[0];
                        shift_reg <= {1'b0, shift_reg[7:1]};
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
            STOP: begin
                if (clk_cnt == CLKS_PER_BIT) begin
                    state <= IDLE;
                    busy <= 0;
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end
        endcase
    end
end
endmodule

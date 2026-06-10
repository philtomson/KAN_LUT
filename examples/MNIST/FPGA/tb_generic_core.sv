// hardware/tb_generic_core.sv
// Testbench to verify the parameterized generic KAN-LUT accelerator with PSRAM backing.

`timescale 1ns/1ps

module tb_generic_core;

  logic clk;
  logic rst;
  logic start;
  logic done;
  
  logic       in_mem_we;
  logic [7:0] in_mem_addr;
  logic [7:0] in_mem_din;
  
  logic [3:0] out_mem_addr;
  logic [7:0] out_mem_dout;
  
  logic        mem_req;
  logic [31:0] mem_addr;
  logic [15:0] mem_rdata;
  logic        mem_rvalid;
  logic        mem_ready;

  // Instantiate the top-level design under test (DUT)
  mnist_generic_top dut (
      .clk(clk),
      .rst(rst),
      .start(start),
      .done(done),
      .in_mem_we(in_mem_we),
      .in_mem_addr(in_mem_addr),
      .in_mem_din(in_mem_din),
      .out_mem_addr(out_mem_addr),
      .out_mem_dout(out_mem_dout),
      .mem_req(mem_req),
      .mem_addr(mem_addr),
      .mem_rdata(mem_rdata),
      .mem_rvalid(mem_rvalid),
      .mem_ready(mem_ready)
  );

  // Mock PSRAM Memory
  logic [15:0] psram_mem [0:4000000];

  // Temporary flat arrays to load the lane files
  logic [13:0] temp_l1_lane_0 [0:802815];
  logic [13:0] temp_l1_lane_1 [0:802815];
  logic [13:0] temp_l1_lane_2 [0:802815];
  logic [13:0] temp_l1_lane_3 [0:802815];

  logic [12:0] temp_l2_lane_0 [0:40959];
  logic [12:0] temp_l2_lane_1 [0:40959];
  logic [12:0] temp_l2_lane_2 [0:40959];
  logic [12:0] temp_l2_lane_3 [0:40959];

  // Load and Map Lane weight files to a single contiguous PSRAM image
  initial begin
    int q, c, x;
    int lane_addr;
    int psram_base;

    // Load Layer 1 Lane Files
    $readmemh("layer1_lane0.mem", temp_l1_lane_0);
    $readmemh("layer1_lane1.mem", temp_l1_lane_1);
    $readmemh("layer1_lane2.mem", temp_l1_lane_2);
    $readmemh("layer1_lane3.mem", temp_l1_lane_3);
    
    // Map Layer 1 Lanes to PSRAM
    for (q = 0; q < 64; q++) begin
      for (c = 0; c < 49; c++) begin
        for (x = 0; x < 256; x++) begin
          lane_addr = (q * 49 + c) * 256 + x;
          psram_base = (q * 196 + c * 4) * 256 + x;
          psram_mem[psram_base + 0*256] = temp_l1_lane_0[lane_addr];
          psram_mem[psram_base + 1*256] = temp_l1_lane_1[lane_addr];
          psram_mem[psram_base + 2*256] = temp_l1_lane_2[lane_addr];
          psram_mem[psram_base + 3*256] = temp_l1_lane_3[lane_addr];
        end
      end
    end

    // Load Layer 2 Lane Files
    $readmemh("layer2_lane0.mem", temp_l2_lane_0);
    $readmemh("layer2_lane1.mem", temp_l2_lane_1);
    $readmemh("layer2_lane2.mem", temp_l2_lane_2);
    $readmemh("layer2_lane3.mem", temp_l2_lane_3);

    // Map Layer 2 Lanes to PSRAM (offset = 3211264)
    for (q = 0; q < 10; q++) begin
      for (c = 0; c < 16; c++) begin
        for (x = 0; x < 256; x++) begin
          lane_addr = (q * 16 + c) * 256 + x;
          psram_base = 3211264 + (q * 64 + c * 4) * 256 + x;
          psram_mem[psram_base + 0*256] = temp_l2_lane_0[lane_addr];
          psram_mem[psram_base + 1*256] = temp_l2_lane_1[lane_addr];
          psram_mem[psram_base + 2*256] = temp_l2_lane_2[lane_addr];
          psram_mem[psram_base + 3*256] = temp_l2_lane_3[lane_addr];
        end
      end
    end
    $display("Mock PSRAM initialization complete.");
  end

  // Cycle-accurate latency simulation: 5-cycle read latency
  assign mem_ready = 1'b1;
  logic [15:0] rdata_queue [0:4];
  logic        rvalid_queue [0:4];

  always @(posedge clk or posedge rst) begin
    if (rst) begin
      for (int i = 0; i < 5; i = i + 1) begin
        rdata_queue[i]  <= '0;
        rvalid_queue[i] <= 1'b0;
      end
    end else begin
      // Shift queue
      for (int i = 0; i < 4; i = i + 1) begin
        rdata_queue[i]  <= rdata_queue[i+1];
        rvalid_queue[i] <= rvalid_queue[i+1];
      end
      
      // Load new request at end of queue
      if (mem_req && mem_ready) begin
        rdata_queue[4]  <= psram_mem[mem_addr];
        rvalid_queue[4] <= 1'b1;
      end else begin
        rdata_queue[4]  <= '0;
        rvalid_queue[4] <= 1'b0;
      end
    end
  end

  assign mem_rdata  = rdata_queue[0];
  assign mem_rvalid = rvalid_queue[0];

  // Clock generation (50 MHz)
  always #10 clk = ~clk;

  // Testbench Variables
  int file_handle;
  int r;
  string line;
  string data_file_path;

  logic [7:0] test_inputs [0:99][0:195];
  logic [7:0] test_outputs [0:99][0:9];

  initial begin
    clk = 0;
    rst = 1;
    start = 0;
    in_mem_we = 0;
    in_mem_addr = 0;
    in_mem_din = 0;
    out_mem_addr = 0;

    // Load test data from file
    if (!$value$plusargs("DATA_FILE=%s", data_file_path)) begin
      data_file_path = "examples/MNIST/FPGA/tb_data.txt";
    end
    file_handle = $fopen(data_file_path, "r");
    if (file_handle == 0) begin
      $display("ERROR: Failed to open %s", data_file_path);
      $finish;
    end

    for (int s = 0; s < 100; s++) begin
      for (int i = 0; i < 196; i++) begin
        logic [7:0] val;
        r = $fscanf(file_handle, "%2h", val);
        test_inputs[s][i] = val;
      end
      for (int o = 0; o < 10; o++) begin
        logic [7:0] val;
        r = $fscanf(file_handle, "%2h", val);
        test_outputs[s][o] = val;
      end
    end
    $fclose(file_handle);
    $display("Successfully loaded 100 test vectors.");

    #100;
    rst = 0;
    #100;

    // Verify all 100 samples
    for (int s = 0; s < 100; s++) begin
      $display("Processing Sample %0d...", s);
      
      // Load input activations
      for (int i = 0; i < 196; i++) begin
        @(posedge clk);
        in_mem_we <= 1;
        in_mem_addr <= i;
        in_mem_din <= test_inputs[s][i];
      end
      @(posedge clk);
      in_mem_we <= 0;
      
      // Trigger execution
      start <= 1;
      @(posedge clk);
      start <= 0;
      
      // Wait for completion
      @(posedge done);
      
      // Read outputs and verify
      for (int o = 0; o < 10; o++) begin
        out_mem_addr = o;
        #1; // allow read propagation
        if (out_mem_dout !== test_outputs[s][o]) begin
          $display("ERROR at Sample %0d, Neuron %0d: Expected %0d (8'h%2h), Got %0d (8'h%2h)", 
                   s, o, test_outputs[s][o], test_outputs[s][o], out_mem_dout, out_mem_dout);
          $finish;
        end
      end
      $display("Sample %0d verified successfully.", s);
    end

    $display("SUCCESS: All 100 samples verified with 100%% bit-accurate parity on PSRAM-backed KAN Core!");
    $finish;
  end

endmodule

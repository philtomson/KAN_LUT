// hardware/tb_generic_core.sv
// Testbench to verify the parameterized generic KAN-LUT accelerator.

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

  // Instantiate the top-level design under test (DUT)
  mnist_generic_top #(
      .L1_INIT_0("layer1_lane0.mem"),
      .L1_INIT_1("layer1_lane1.mem"),
      .L1_INIT_2("layer1_lane2.mem"),
      .L1_INIT_3("layer1_lane3.mem"),
      .L2_INIT_0("layer2_lane0.mem"),
      .L2_INIT_1("layer2_lane1.mem"),
      .L2_INIT_2("layer2_lane2.mem"),
      .L2_INIT_3("layer2_lane3.mem")
  ) dut (
      .clk(clk),
      .rst(rst),
      .start(start),
      .done(done),
      .in_mem_we(in_mem_we),
      .in_mem_addr(in_mem_addr),
      .in_mem_din(in_mem_din),
      .out_mem_addr(out_mem_addr),
      .out_mem_dout(out_mem_dout)
  );

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

    // Load test data from file (support plusargs)
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

    $display("SUCCESS: All 100 samples verified with 100%% bit-accurate parity on Generic KAN Core!");
    $finish;
  end

endmodule

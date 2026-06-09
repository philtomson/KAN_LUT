// examples/MNIST/FPGA/tb_mnist_kan.sv
`timescale 1ns/1ps

module tb_mnist_kan;

  logic clk;
  logic rst;
  logic [7:0] in_val [0:195];
  logic [7:0] out_val [0:9];

  // Instantiate the KAN Top module
  mnist_kan_top uut (
      .clk(clk),
      .rst(rst),
      .in_val(in_val),
      .out_val(out_val)
  );

  // Clock generation (100MHz -> 10ns period)
  always begin
    #5 clk = ~clk;
  end

  // Array to hold test data
  // 100 samples. Each line has: 196 bytes of inputs, space, 10 bytes of outputs.
  // 196 * 2 hex chars = 392 characters for inputs.
  // 10 * 2 hex chars = 20 characters for outputs.
  logic [7:0] test_inputs [0:99][0:195];
  logic [7:0] test_outputs [0:99][0:9];

  int file_handle;
  int r;
  string line;
  string data_file_path;

  initial begin
    clk = 0;
    rst = 1;
    
    // Clear inputs
    for (int i = 0; i < 196; i++) begin
      in_val[i] = 8'h00;
    end

    // Load test data from file
    if (!$value$plusargs("DATA_FILE=%s", data_file_path)) begin
      data_file_path = "tb_data.txt";
    end
    file_handle = $fopen(data_file_path, "r");
    if (file_handle == 0) begin
      $display("ERROR: Failed to open %s", data_file_path);
      $finish;
    end

    for (int s = 0; s < 100; s++) begin
      // Read 196 inputs (in hex format)
      for (int i = 0; i < 196; i++) begin
        logic [7:0] val;
        r = $fscanf(file_handle, "%2h", val);
        test_inputs[s][i] = val;
      end
      
      // Read space and then 10 outputs
      for (int o = 0; o < 10; o++) begin
        logic [7:0] val;
        r = $fscanf(file_handle, "%2h", val);
        test_outputs[s][o] = val;
      end
    end
    $fclose(file_handle);
    $display("Successfully loaded 100 test vectors.");
    $display("Debug input load: test_inputs[0][0] = %2h, [0][1] = %2h, [0][195] = %2h", test_inputs[0][0], test_inputs[0][1], test_inputs[0][195]);

    // Reset pulse
    #20;
    rst = 0;
    #10;

    $display("Starting pipelined verification...");
    
    // Process samples
    // Pipeline latency is 15 cycles: Layer 1 (9 cycles) + Layer 2 (6 cycles)
    // We feed inputs from cycle 0 to 99.
    // We check outputs from cycle 15 to 114.
    fork
      // Thread 1: Feed inputs
      begin
        for (int s = 0; s < 100; s++) begin
          @(posedge clk);
          for (int i = 0; i < 196; i++) begin
            in_val[i] = test_inputs[s][i];
          end
        end
        // Clear inputs after feeding is done
        @(posedge clk);
        for (int i = 0; i < 196; i++) begin
          in_val[i] = 8'h00;
        end
      end

      // Thread 2: Monitor and verify outputs
      begin
        // Wait 16 cycles for the first result
        for (int c = 0; c < 16; c++) begin
          @(posedge clk);
        end
        
        // Now check results
        for (int s = 0; s < 100; s++) begin
          #1; // Wait small delay after clock edge to let signals stabilize
          for (int o = 0; o < 10; o++) begin
            if (out_val[o] !== test_outputs[s][o]) begin
              $display("ERROR at sample %0d, output %0d: Expected %0d (8'h%2h), Got %0d (8'h%2h)", 
                       s, o, test_outputs[s][o], test_outputs[s][o], out_val[o], out_val[o]);
              $finish;
            end
          end
          $display("Sample %0d verified successfully.", s);
          @(posedge clk);
        end
        $display("SUCCESS: All 100 samples verified with 100%% bit-accurate parity!");
        $finish;
      end
    join
  end

endmodule

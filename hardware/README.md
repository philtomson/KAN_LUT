# Reusable KAN Hardware IP Cores

This directory contains the highly reusable, general-purpose hardware IP cores and peripheral controllers. These components are independent of any specific machine learning task (such as MNIST) and are designed to be parameterized for different architectures and datasets.

---

## Directory Overview

*   **`kan_generic_core.sv`**: The core parameterized KAN processor. It performs parallel lookups on 1D edge spline functions and runs the summation logic.
*   **`kan_bram_bank.sv`**: Dual-port Block RAM cache banks used by the KAN core for fast weight lookup.
*   **`sdram.v`**: A byte-addressable SDR SDRAM controller configured for the internal 64Mbit embedded memory of the Gowin GW2AR-18C FPGA (on the Sipeed Tang Nano 20K).
*   **`uart_rx.v` & `uart_tx.v`**: Parameterized high-speed UART transceiver modules.
*   **`gowin_rpll/`**: Phase-locked loop configuration setting the main system clock to **54 MHz** (used for high-speed UART sampling and SDRAM access).

---

## How to Reuse These Components

To instantiate these modules in your own application, add this directory to your include paths and instantiate the cores with custom parameters.

### Parameterizing `kan_generic_core`
```systemverilog
kan_generic_core #(
    .INPUT_DIM(2),          // Dimension of input vector
    .OUTPUT_DIM(8),         // Dimension of output vector
    .PARALLELISM(4),        // Number of execution lanes running in parallel
    .LUT_DEPTH(256),        // Number of 1D spline points per edge function
    .DATA_WIDTH(14),        // Bit-width of weights and activations
    .FRACTIONAL_BITS(4)     // Fixed-point precision
) core_inst (
    .clk(clk),
    .rst(rst),
    .start(start),
    .done(done),
    ...
);
```

---

## Application Examples
For concrete deployments and simulation setups of these cores, refer to:
*   [examples/MNIST/FPGA](file:///home/phil/devel/FPGA/KAN_LUT/examples/MNIST/FPGA): Includes the 2-layer MNIST generic top wrapper, interactive Verilator canvas simulation, and Sipeed Tang Nano 20K physical integration.

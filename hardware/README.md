# KAN LUT FPGA Hardware Implementation & Deployment

This directory contains the SystemVerilog RTL implementation, simulation testbenches, and hardware synthesis wrappers for deploying the Kolmogorov-Arnold Network (KAN) generic core onto the **Sipeed Tang Nano 20K FPGA** (based on the Gowin GW2AR-18C device).

---

## Directory Overview

*   **RTL Core**:
    *   `kan_generic_core.sv`: The core parameterized KAN processor.
    *   `kan_bram_bank.sv`: Cache memory block used for coefficient lookups.
    *   `mnist_generic_top.sv`: Top-level model linking the core to memory interfaces for digit classification.
*   **Tang Nano 20K Wrapper**:
    *   `nano20k_top.sv`: Top-level hardware wrapper managing clocking, UART communication, memory arbitration, and the KAN core.
    *   `sdram.v`: Open-source SDRAM controller managing the internal 64Mbit embedded SDR SDRAM on the GW2AR-18C.
    *   `gowin_rpll/`: Phase-locked loop configuration setting the main system clock to **54 MHz** (and providing a phase-shifted clock for the SDRAM).
    *   `uart_rx.v` & `uart_tx.v`: Custom UART transceiver modules operating at **2 Mbaud**.
    *   `nano20k.cst`: Physical pin constraints mapping the system clock, UART RX/TX, buttons, and status LEDs.
*   **Simulation & Testing**:
    *   `tb_generic_core.sv`: Standard testbench for verifying the core functionality.
    *   `sim_interactive.cpp`: Verilator-based interactive simulation with SDL2 graphics support.
    *   `test_nano20k.py`: Python host script to load weight JSONs, serialize them, stream them to the FPGA, and run test inferece.
    *   `Makefile`: Automates synthesis, place-and-route, and simulation commands.

---

## 1. Simulation & Interactive Verification

### Standard Testbench
To compile and run the standard IVerilog simulation testbench:
```bash
make compile
make run
```

### Interactive Verilator + SDL2 Simulation
To launch the interactive SDL2-based simulation where you can draw digits in real-time and verify classification:
```bash
make sim_interactive
```

---

## 2. FPGA Synthesis & Bitstream Generation

The hardware pipeline utilizes the open-source **Apicula** / **Yosys** / **nextpnr-himbaechel** toolchain to target the Gowin GW2AR-18C device.

To run synthesis, place-and-route, and pack the bitstream:
```bash
make nano20k
```
This produces `pack.fs` (the binary bitstream).

---

## 3. Flashing the Bitstream

To flash the synthesized bitstream (`pack.fs`) onto the Tang Nano 20K using `openFPGALoader`:

```bash
openFPGALoader -b tangnano20k pack.fs
```

---

## 4. Hardware Inference & Weight Streaming

The FPGA bootloader runs a serial interface at **2,000,000 baud** (2 Mbaud). When the board boots up, it enters a `S_LOAD_WEIGHT` state expecting the serialized weights, and then transitions to the `S_INFER_WAIT` state.

To stream the trained weights and execute real-time inference on MNIST test images:

1.  Make sure you have `pyserial` installed:
    ```bash
    pip install pyserial
    ```
2.  Run the Python host script targeting the serial port of your Tang Nano 20K (typically `/dev/ttyUSB1`):
    ```bash
    python test_nano20k.py /dev/ttyUSB1
    ```

The host script will:
*   Convert the floating/quantized KAN Layer JSON files into a flat, 16-bit binary payload (L1 + L2 weights).
*   Upload the 3.37 MB payload to the SDRAM via UART (takes approx. 16 seconds).
*   Iteratively stream 196-byte MNIST test samples, wait for the classification byte response, and report accuracy.

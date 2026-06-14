// tb_mnist_kan_mixed.cpp - Verilator C++ Testbench for unrolled mixed-precision KAN
#include <verilated.h>
#include "Vmnist_kan_top.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdint>
#include <cmath>

static Vmnist_kan_top *dut;

void tick() {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

void dut_reset() {
    dut->rst = 1;
    for (int w = 0; w < 7; w++) {
        dut->in_val[w] = 0;
    }
    for (int i = 0; i < 10; i++) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
    }
    dut->rst = 0;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vmnist_kan_top;

    std::string data_file_path = "tb_data_mixed.txt";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg.find("+DATA_FILE=") == 0) {
            data_file_path = arg.substr(11);
        }
    }

    std::ifstream file(data_file_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Failed to open " << data_file_path << std::endl;
        return 1;
    }

    std::cout << "Successfully loaded mixed-precision test vectors from " << data_file_path << std::endl;
    std::cout << "Starting pipelined verification for mixed-precision KAN..." << std::endl;

    // We store inputs and expected outputs for all 100 samples
    uint8_t test_inputs[100][196];
    uint8_t test_outputs[100][10];

    for (int s = 0; s < 100; s++) {
        std::string input_str;
        if (!(file >> input_str)) {
            std::cerr << "ERROR: Failed to read input string for sample " << s << std::endl;
            return 1;
        }
        if (input_str.length() < 392) {
            std::cerr << "ERROR: Input string for sample " << s << " is too short (" << input_str.length() << " chars)" << std::endl;
            return 1;
        }
        for (int i = 0; i < 196; i++) {
            std::string byte_str = input_str.substr(2 * i, 2);
            test_inputs[s][i] = (uint8_t)std::stoul(byte_str, nullptr, 16);
        }

        std::string output_str;
        if (!(file >> output_str)) {
            std::cerr << "ERROR: Failed to read output string for sample " << s << std::endl;
            return 1;
        }
        if (output_str.length() < 20) {
            std::cerr << "ERROR: Output string for sample " << s << " is too short (" << output_str.length() << " chars)" << std::endl;
            return 1;
        }
        for (int o = 0; o < 10; o++) {
            std::string byte_str = output_str.substr(2 * o, 2);
            test_outputs[s][o] = (uint8_t)std::stoul(byte_str, nullptr, 16);
        }
    }
    file.close();

    // Verify each sample. Since it is pipelined, we can feed one sample per cycle!
    // Total latency is 15 cycles.
    // Feed inputs: cycle 0 to 99.
    // Check outputs: cycle 15 to 114.
    dut_reset();

    uint8_t fed_inputs[120][196];
    std::memset(fed_inputs, 0, sizeof(fed_inputs));

    int passed_samples = 0;

    for (int cycle = 0; cycle < 120; cycle++) {
        // Feed input if within 100 samples
        if (cycle < 100) {
            for (int w = 0; w < 7; w++) {
                dut->in_val[w] = 0;
            }
            for (int i = 0; i < 196; i++) {
                int word_idx = i / 32;
                int bit_idx = i % 32;
                uint32_t bit = (test_inputs[cycle][i] > 0) ? 1 : 0;
                dut->in_val[word_idx] |= (bit << bit_idx);
            }
        } else {
            for (int w = 0; w < 7; w++) {
                dut->in_val[w] = 0;
            }
        }

        tick();

        // Check outputs if we have reached latency threshold
        uint64_t raw_out = dut->out_val;
        if (cycle < 10) {
            std::cout << "Cycle " << cycle << " raw_out: 0x" << std::hex << raw_out << std::dec << std::endl;
        }
        if (cycle >= 1) {
            int sample_idx = cycle - 1;
            if (sample_idx < 100) {
                bool sample_ok = true;
                for (int o = 0; o < 10; o++) {
                    int64_t score_6bit = (raw_out >> (o * 6)) & 0x3F;
                    if (score_6bit & 0x20) {
                        score_6bit |= ~0x3F; // Sign extend got score
                    }
                    int got_score = (int)score_6bit;

                    int expected_val = test_outputs[sample_idx][o] & 0x3F;
                    if (expected_val & 0x20) {
                        expected_val |= ~0x3F; // Sign extend expected score
                    }
                    int expected_score = expected_val;
                    
                    if (got_score != expected_score) {
                        std::cerr << "ERROR at sample " << sample_idx << ", output " << o
                                  << ": Expected " << expected_score << ", Got " << got_score << std::endl;
                        sample_ok = false;
                        return 1;
                    }
                }
                if (sample_ok) {
                    std::cout << "Sample " << sample_idx << " verified successfully." << std::endl;
                    passed_samples++;
                }
            }
        }
    }

    if (passed_samples == 100) {
        std::cout << "SUCCESS: All 100 mixed-precision samples verified with 100% bit-accurate parity!" << std::endl;
    } else {
        std::cerr << "ERROR: Only verified " << passed_samples << "/100 samples." << std::endl;
        return 1;
    }

    dut->final();
    delete dut;
    return 0;
}

// sim_interactive.cpp - Interactive KAN-LUT digit classifier simulation
//
// Draw a digit on the 28×28 canvas (left-click drag, right-click erases).
// Click CLASSIFY or press ENTER to run inference through the KAN verilated model.
//
// Keys:  ENTER/SPACE = classify   C/BKSP = clear   ESC = quit
//        +/- = grow/shrink brush

#include <SDL2/SDL.h>
#include <verilated.h>
#include "Vmnist_generic_top.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>

// ── Window layout ─────────────────────────────────────────────────────────────
static const int CELL   = 18;           // screen pixels per canvas cell
static const int GRID   = 28;           // canvas is 28×28
static const int CSIZE  = CELL * GRID;  // 504
static const int PANEL  = 210;
static const int WIN_W  = CSIZE + PANEL;
static const int WIN_H  = CSIZE;

// ── Drawing state ─────────────────────────────────────────────────────────────
static uint8_t canvas[GRID * GRID];     // 0=empty, 255=drawn
static int     brush = 2;              // brush side-length

// ── Test sample bank (from tb_data.txt) ──────────────────────────────────────
static uint8_t tb_inputs[100][196];    // pre-quantized 14×14, column-major
static uint8_t tb_expected_out[100][10]; // expected output scores
static int     tb_expected_digit[100]; // expected argmax
static int     tb_count = 0;           // how many loaded
static int     cur_sample = -1;        // -1 = freehand mode

// ── 5x7 bitmap font for digits 0-9 ──────────────────────────────────────────
static const uint8_t FONT[10][7] = {
    {0x0E,0x11,0x13,0x15,0x19,0x11,0x0E}, // 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
    {0x0E,0x11,0x01,0x06,0x08,0x10,0x1F}, // 2
    {0x0E,0x11,0x01,0x06,0x01,0x11,0x0E}, // 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
    {0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E}, // 5
    {0x06,0x08,0x10,0x1E,0x11,0x11,0x0E}, // 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
    {0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C}, // 9
};

static void draw_glyph(SDL_Renderer *ren, int d, int x, int y, int s) {
    for (int row = 0; row < 7; row++) {
        for (int col = 0; col < 5; col++) {
            if ((FONT[d][row] >> (4 - col)) & 1) {
                SDL_Rect r = {x + col*s, y + row*s, s, s};
                SDL_RenderFillRect(ren, &r);
            }
        }
    }
}

// ── Verilator DUT & PSRAM ─────────────────────────────────────────────────────
static Vmnist_generic_top *dut;
static uint16_t psram_mem[2000000];

// Latency Queue for C++ Cycle-accurate PSRAM Simulation (5-cycle latency)
static uint16_t rdata_queue[5];
static bool     rvalid_queue[5];

static void tick() {
    // This function models one clock cycle, exactly matching tb_generic_core.sv:
    //
    //   always @(posedge clk) begin
    //     for (i=0; i<4; i++) rdata_queue[i] <= rdata_queue[i+1];
    //     if (mem_req) rdata_queue[4] <= psram_mem[mem_addr];
    //     else         rdata_queue[4] <= 0;
    //   end
    //   assign mem_rdata  = rdata_queue[0];
    //   assign mem_rvalid = rvalid_queue[0];
    //
    // Step 1: present current queue outputs combinationally to DUT, drive clk low
    dut->mem_rdata  = rdata_queue[0];
    dut->mem_rvalid = rvalid_queue[0];
    dut->mem_ready  = 1;
    dut->clk = 0;
    dut->eval();

    // Step 2: sample DUT outputs that the always-block would see at posedge
    bool    req  = (bool)dut->mem_req;
    uint32_t addr = dut->mem_addr;

    // Step 3: raise clock — DUT's registered logic latches
    dut->clk = 1;
    dut->eval();

    // Step 4: model the always @(posedge clk) queue shift (happens at posedge)
    for (int i = 0; i < 4; i++) {
        rdata_queue[i]  = rdata_queue[i + 1];
        rvalid_queue[i] = rvalid_queue[i + 1];
    }
    if (req && addr < 2000000) {
        rdata_queue[4]  = psram_mem[addr];
        rvalid_queue[4] = true;
    } else {
        rdata_queue[4]  = 0;
        rvalid_queue[4] = false;
    }
}

static void dut_reset() {
    dut->rst = 1;
    dut->start = 0;
    dut->in_mem_we = 0;
    dut->in_mem_addr = 0;
    dut->in_mem_din = 0;
    dut->out_mem_addr = 0;
    
    // Clock reset
    for (int i = 0; i < 10; i++) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
    }
    dut->rst = 0;
    
    // Reset PSRAM queues
    for (int i = 0; i < 5; i++) {
        rdata_queue[i] = 0;
        rvalid_queue[i] = false;
    }
}

// Helper to load hex memory files
bool load_mem_file(const char *filepath, uint16_t *dest, int expected_size) {
    FILE *f = fopen(filepath, "r");
    if (!f) {
        printf("ERROR: Could not open %s\n", filepath);
        return false;
    }
    char line[128];
    int count = 0;
    while (fgets(line, sizeof(line), f) && count < expected_size) {
        // Skip comments or empty lines
        if (line[0] == '/' || line[0] == '\n' || line[0] == '\r') continue;
        unsigned int val;
        if (sscanf(line, "%x", &val) == 1) {
            dest[count++] = (uint16_t)val;
        }
    }
    fclose(f);
    if (count != expected_size) {
        printf("WARNING: Loaded only %d/%d values from %s\n", count, expected_size, filepath);
    }
    return true;
}

// Unified PSRAM loading and mapping
bool init_psram() {
    memset(psram_mem, 0, sizeof(psram_mem));
    
    // Temporary flat arrays for lanes
    std::vector<uint16_t> temp_l1_lane_0(401408, 0);
    std::vector<uint16_t> temp_l1_lane_1(401408, 0);
    std::vector<uint16_t> temp_l1_lane_2(401408, 0);
    std::vector<uint16_t> temp_l1_lane_3(401408, 0);

    std::vector<uint16_t> temp_l2_lane_0(20480, 0);
    std::vector<uint16_t> temp_l2_lane_1(20480, 0);
    std::vector<uint16_t> temp_l2_lane_2(20480, 0);
    std::vector<uint16_t> temp_l2_lane_3(20480, 0);

    printf("Loading Layer 1 Lanes...\n");
    if (!load_mem_file("layer1_lane0.mem", temp_l1_lane_0.data(), 401408)) return false;
    if (!load_mem_file("layer1_lane1.mem", temp_l1_lane_1.data(), 401408)) return false;
    if (!load_mem_file("layer1_lane2.mem", temp_l1_lane_2.data(), 401408)) return false;
    if (!load_mem_file("layer1_lane3.mem", temp_l1_lane_3.data(), 401408)) return false;

    printf("Loading Layer 2 Lanes...\n");
    if (!load_mem_file("layer2_lane0.mem", temp_l2_lane_0.data(), 20480)) return false;
    if (!load_mem_file("layer2_lane1.mem", temp_l2_lane_1.data(), 20480)) return false;
    if (!load_mem_file("layer2_lane2.mem", temp_l2_lane_2.data(), 20480)) return false;
    if (!load_mem_file("layer2_lane3.mem", temp_l2_lane_3.data(), 20480)) return false;

    // Map Layer 1 Lanes to PSRAM
    for (int q = 0; q < 32; q++) {
        for (int c = 0; c < 49; c++) {
            for (int x = 0; x < 256; x++) {
                int lane_addr = (q * 49 + c) * 256 + x;
                int psram_base = (q * 196 + c * 4) * 256 + x;
                psram_mem[psram_base + 0*256] = temp_l1_lane_0[lane_addr];
                psram_mem[psram_base + 1*256] = temp_l1_lane_1[lane_addr];
                psram_mem[psram_base + 2*256] = temp_l1_lane_2[lane_addr];
                psram_mem[psram_base + 3*256] = temp_l1_lane_3[lane_addr];
            }
        }
    }

    // Map Layer 2 Lanes to PSRAM (offset = 1605632)
    for (int q = 0; q < 10; q++) {
        for (int c = 0; c < 8; c++) {
            for (int x = 0; x < 256; x++) {
                int lane_addr = (q * 8 + c) * 256 + x;
                int psram_base = 1605632 + (q * 32 + c * 4) * 256 + x;
                psram_mem[psram_base + 0*256] = temp_l2_lane_0[lane_addr];
                psram_mem[psram_base + 1*256] = temp_l2_lane_1[lane_addr];
                psram_mem[psram_base + 2*256] = temp_l2_lane_2[lane_addr];
                psram_mem[psram_base + 3*256] = temp_l2_lane_3[lane_addr];
            }
        }
    }

    printf("PSRAM initialized successfully.\n");
    return true;
}

// ── Simple 2×2 average pool, filling dst196 in Julia column-major order ──────
// generate_test_vectors.jl does:
//   x_down[i,j,n] = avg of 4 pixels        (Julia 1-indexed)
//   X = reshape(x_down, 196, :)             (column-major: i varies fastest)
//   for p in 1:196: write X[p, b]          (p-1 → row=(p-1)%14, col=(p-1)/14)
//
// So hardware address k (0-indexed) = pixel at row k%14, col k/14 of 14×14 image.
// We fill dst196[k] = avg of canvas 2×2 block at that (row, col).
// ── Load a known test sample onto the 28×28 canvas (2× upscale) ─────────────
static void load_sample_to_canvas(int idx) {
    if (idx < 0 || idx >= tb_count) return;
    cur_sample = idx;
    memset(canvas, 0, sizeof(canvas));
    // tb_inputs[idx][k] is column-major: k = col*14 + row, value in [128,255]
    // Reverse quantization: canvas_val ≈ (q - 127.5) * 16
    // Upscale 14×14 → 28×28 by repeating each pixel into a 2×2 block.
    // Julia MNIST: features[x,y,n] — first dim is x (horizontal=col), second is y (vertical=row).
    // Column-major reshape: k%14 = img_col (x), k/14 = img_row (y).
    for (int k = 0; k < 196; k++) {
        int img_col = k % 14;  // x / horizontal
        int img_row = k / 14;  // y / vertical
        uint8_t q   = tb_inputs[idx][k];
        float   v_f = ((float)q - 127.5f) * 16.0f;
        int     v   = (int)v_f;
        if (v < 0) v = 0;
        if (v > 255) v = 255;
        for (int dr = 0; dr < 2; dr++)
            for (int dc = 0; dc < 2; dc++)
                canvas[(img_row*2+dr)*GRID + (img_col*2+dc)] = (uint8_t)v;
    }
}

static void simple_pool_canvas(const uint8_t *src28, uint8_t *dst196) {
    // Julia MNIST: features[x,y,n] — first dim is x (horizontal=col), second is y (vertical=row).
    // Column-major reshape: k%14 = img_col (x/horizontal), k/14 = img_row (y/vertical).
    for (int k = 0; k < 196; k++) {
        int img_col = k % 14;  // x / horizontal
        int img_row = k / 14;  // y / vertical
        int r0 = 2 * img_row, c0 = 2 * img_col;
        int sum = (int)src28[r0     * GRID + c0]
                + (int)src28[r0     * GRID + c0 + 1]
                + (int)src28[(r0+1) * GRID + c0]
                + (int)src28[(r0+1) * GRID + c0 + 1];
        dst196[k] = (uint8_t)(sum / 4);
    }
}

// ── Rendering layout ─────────────────────────────────────────────────────────
static const SDL_Rect CLS_BTN = {CSIZE+10,  10, PANEL-20, 44};
static const SDL_Rect CLR_BTN = {CSIZE+10,  58, PANEL-20, 38};

static void render(SDL_Renderer *ren, const std::vector<int>& scores, int result, bool busy) {
    SDL_SetRenderDrawColor(ren, 20, 20, 20, 255);
    SDL_RenderClear(ren);

    // Canvas background
    SDL_SetRenderDrawColor(ren, 32, 32, 36, 255);
    SDL_Rect cvbg = {0, 0, CSIZE, WIN_H};
    SDL_RenderFillRect(ren, &cvbg);

    // Canvas cells
    for (int r = 0; r < GRID; r++) {
        for (int c = 0; c < GRID; c++) {
            uint8_t v = canvas[r*GRID+c];
            SDL_SetRenderDrawColor(ren, v, v, v, 255);
            SDL_Rect cell = {c*CELL+1, r*CELL+1, CELL-2, CELL-2};
            SDL_RenderFillRect(ren, &cell);
        }
    }

    // Grid lines
    SDL_SetRenderDrawColor(ren, 52, 52, 56, 255);
    for (int i = 0; i <= GRID; i++) {
        SDL_RenderDrawLine(ren, i*CELL,  0,     i*CELL, CSIZE);
        SDL_RenderDrawLine(ren, 0,       i*CELL, CSIZE,  i*CELL);
    }

    // Right panel
    SDL_SetRenderDrawColor(ren, 38, 38, 48, 255);
    SDL_Rect panel = {CSIZE, 0, PANEL, WIN_H};
    SDL_RenderFillRect(ren, &panel);

    // CLASSIFY button
    SDL_SetRenderDrawColor(ren, busy ? 50 : 46, busy ? 100 : 184, busy ? 50 : 114, 255);
    SDL_RenderFillRect(ren, &CLS_BTN);
    // play arrow glyph in button
    SDL_SetRenderDrawColor(ren, 220, 255, 220, 255);
    int bx = CLS_BTN.x + CLS_BTN.w/2 - 8, by = CLS_BTN.y + CLS_BTN.h/2;
    for (int i = 0; i < 12; i++) {
        SDL_RenderDrawLine(ren, bx, by-i, bx+i, by);
        SDL_RenderDrawLine(ren, bx, by+i, bx+i, by);
    }

    // CLEAR button
    SDL_SetRenderDrawColor(ren, 186, 73, 73, 255);
    SDL_RenderFillRect(ren, &CLR_BTN);
    // X glyph in clear button
    SDL_SetRenderDrawColor(ren, 255, 220, 220, 255);
    int cx1 = CLR_BTN.x+20, cy1 = CLR_BTN.y+10;
    int cx2 = CLR_BTN.x+CLR_BTN.w-20, cy2 = CLR_BTN.y+CLR_BTN.h-10;
    SDL_RenderDrawLine(ren, cx1, cy1, cx2, cy2);
    SDL_RenderDrawLine(ren, cx2, cy1, cx1, cy2);

    // Separator line
    SDL_SetRenderDrawColor(ren, 70, 70, 80, 255);
    SDL_RenderDrawLine(ren, CSIZE, 102, WIN_W, 102);

    // Sample indicator (test mode vs freehand)
    if (cur_sample >= 0 && cur_sample < tb_count) {
        // Show sample index and expected digit
        int exp = tb_expected_digit[cur_sample];
        SDL_SetRenderDrawColor(ren, 80, 60, 120, 255);
        SDL_Rect sbox = {CSIZE+6, 104, PANEL-12, 22};
        SDL_RenderFillRect(ren, &sbox);
        // Draw "E:" label glyph
        SDL_SetRenderDrawColor(ren, 200, 180, 255, 255);
        // Draw expected digit large
        draw_glyph(ren, exp, CSIZE + PANEL/2 - 5, 106, 3);
    }
    SDL_SetRenderDrawColor(ren, 70, 70, 80, 255);
    SDL_RenderDrawLine(ren, CSIZE, 130, WIN_W, 130);

    // Find min and max scores for relative visualization
    int min_score = 999999;
    int max_score = -999999;
    for (int s : scores) {
        if (s < min_score) min_score = s;
        if (s > max_score) max_score = s;
    }
    int score_range = max_score - min_score;
    if (score_range <= 0) score_range = 1;

    // Draw Digit Progress Bars (10 rows)
    const int BAR_H = 34, BAR_Y0 = 132;
    for (int d = 0; d < 10; d++) {
        bool active = (result == d);
        int y = BAR_Y0 + d * BAR_H;

        // Draw relative bar width
        float fraction = (float)(scores[d] - min_score) / (float)score_range;
        int bar_w = (int)(fraction * (PANEL - 40));
        if (bar_w < 4) bar_w = 4;

        // Bar background card
        SDL_SetRenderDrawColor(ren, 48, 48, 58, 255);
        SDL_Rect bg_card = {CSIZE+6, y+2, PANEL-12, BAR_H-4};
        SDL_RenderFillRect(ren, &bg_card);

        // Score fill bar (gradient style colors)
        if (active) {
            SDL_SetRenderDrawColor(ren, 46, 184, 114, 255); // Green for winner
        } else {
            SDL_SetRenderDrawColor(ren, 68, 115, 196, 255);  // Blue for others
        }
        SDL_Rect fill_rect = {CSIZE+32, y+10, bar_w, BAR_H-20};
        SDL_RenderFillRect(ren, &fill_rect);

        // Digit glyph on left
        SDL_SetRenderDrawColor(ren, active ? 255 : 180, active ? 255 : 180, active ? 255 : 200, 255);
        draw_glyph(ren, d, CSIZE+12, y + (BAR_H - 14)/2, 2);
    }

    if (busy) {
        SDL_SetRenderDrawBlendMode(ren, SDL_BLENDMODE_BLEND);
        SDL_SetRenderDrawColor(ren, 0, 0, 0, 160);
        SDL_Rect overlay = {0, 0, WIN_W, WIN_H};
        SDL_RenderFillRect(ren, &overlay);
    }

    SDL_RenderPresent(ren);
}

// Painting brush utility
static void paint(int col, int row, bool erase) {
    float sigma = brush * 0.30f + 0.25f;
    int   ext   = (int)(sigma * 2.5f) + 1;
    for (int dr = -ext; dr <= ext; dr++) {
        for (int dc = -ext; dc <= ext; dc++) {
            int r = row + dr, c = col + dc;
            if (r < 0 || r >= GRID || c < 0 || c >= GRID) continue;
            if (erase) {
                canvas[r*GRID+c] = 0;
            } else {
                float val_f = 255.0f * expf(-(dr*dr + dc*dc) / (2.0f * sigma * sigma));
                uint8_t val = (uint8_t)(val_f > 254.0f ? 255 : val_f);
                if (val > canvas[r*GRID+c]) canvas[r*GRID+c] = val;
            }
        }
    }
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vmnist_generic_top;
    
    if (!init_psram()) {
        fprintf(stderr, "Failed to load/map weights to mock PSRAM.\n");
        return 1;
    }

    // ── Self-test: run first 5 samples from tb_data.txt to validate PSRAM + DUT ──
    {
        FILE *f = fopen("tb_data.txt", "r");
        if (!f) f = fopen("../examples/MNIST/FPGA/tb_data.txt", "r");
        if (!f) f = fopen("examples/MNIST/FPGA/tb_data.txt", "r");
        if (f) {
            printf("=== SELF-TEST: validating against tb_data.txt ===\n");
            int passed = 0;
            for (int s = 0; s < 5; s++) {
                uint8_t tb_inputs[196];
                uint8_t tb_expected[10];
                // Read 196 input hex bytes
                for (int i = 0; i < 196; i++) {
                    unsigned int v; fscanf(f, "%2x", &v);
                    tb_inputs[i] = (uint8_t)v;
                }
                fgetc(f); // space
                // Read 10 output hex bytes
                for (int o = 0; o < 10; o++) {
                    unsigned int v; fscanf(f, "%2x", &v);
                    tb_expected[o] = (uint8_t)v;
                }
                fgetc(f); // newline

                // Run inference with known-good inputs (already quantized, no transpose)
                dut_reset();
                for (int i = 0; i < 196; i++) {
                    dut->clk = 0; dut->in_mem_we = 1;
                    dut->in_mem_addr = i; dut->in_mem_din = tb_inputs[i];
                    dut->eval();
                    dut->clk = 1; dut->eval();
                }
                dut->in_mem_we = 0;
                dut->clk = 0; dut->start = 1; dut->eval();
                dut->clk = 1; dut->eval();
                dut->clk = 0; dut->start = 0; dut->eval();

                int cycles = 0;
                while (!dut->done && cycles < 300000) { tick(); cycles++; }

                // Read outputs
                int hw_argmax = 0, hw_max = -999999;
                int ref_argmax = 0, ref_max = -999999;
                printf("  Sample %d: hw=[", s);
                for (int o = 0; o < 10; o++) {
                    dut->out_mem_addr = o; dut->eval();
                    int hw_score = (uint8_t)dut->out_mem_dout;
                    int ref_score = (uint8_t)tb_expected[o];
                    if (hw_score > hw_max)  { hw_max = hw_score;  hw_argmax = o; }
                    if (ref_score > ref_max){ ref_max = ref_score; ref_argmax = o; }
                    printf("%d", hw_score); if(o<9) printf(",");
                }
                printf("] -> pred=%d  ref=%d %s\n",
                    hw_argmax, ref_argmax,
                    hw_argmax == ref_argmax ? "OK" : "FAIL");
                if (hw_argmax == ref_argmax) passed++;
            }
            printf("=== Self-test: %d/5 correct ===\n\n", passed);
            fclose(f);
        } else {
            printf("(self-test skipped: tb_data.txt not found)\n");
        }
    }

    // Load test sample bank from tb_data.txt for ground-truth validation mode
    {
        FILE *f2 = fopen("tb_data.txt", "r");
        if (!f2) f2 = fopen("../examples/MNIST/FPGA/tb_data.txt", "r");
        if (!f2) f2 = fopen("examples/MNIST/FPGA/tb_data.txt", "r");
        if (f2) {
            for (int s = 0; s < 100; s++) {
                for (int i = 0; i < 196; i++) {
                    unsigned int v; fscanf(f2, "%2x", &v);
                    tb_inputs[s][i] = (uint8_t)v;
                }
                fgetc(f2); // space
                int ref_max = -999, ref_argmax = 0;
                for (int o = 0; o < 10; o++) {
                    unsigned int v; fscanf(f2, "%2x", &v);
                    tb_expected_out[s][o] = (uint8_t)v;
                    if ((int)(uint8_t)v > ref_max) { ref_max = (int)(uint8_t)v; ref_argmax = o; }
                }
                fgetc(f2); // newline
                tb_expected_digit[s] = ref_argmax;
                tb_count++;
            }
            fclose(f2);
            printf("Loaded %d test samples from tb_data.txt\n", tb_count);
        }
    }

    dut_reset();

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow(
        "KAN-LUT Simulator  |  ENTER=classify  N/P=next/prev sample  C=clear  +/-=brush",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIN_W, WIN_H, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

    // Start with first known test sample pre-drawn
    memset(canvas, 0, sizeof(canvas));
    if (tb_count > 0) load_sample_to_canvas(0);
    std::vector<int> scores(10, 0);
    int  result   = -1;
    bool running  = true;
    bool mouse_dn = false;
    bool erasing  = false;

    while (running) {
        SDL_Event ev;
        while (SDL_PollEvent(&ev)) {
            switch (ev.type) {
            case SDL_QUIT:
                running = false;
                break;

            case SDL_KEYDOWN:
                switch (ev.key.keysym.sym) {
                case SDLK_RETURN: case SDLK_SPACE: {
                    render(ren, scores, result, true);
                    
                    // Downsample canvas to 196 values in column-major hardware order
                    uint8_t inputs196[196];
                    simple_pool_canvas(canvas, inputs196);

                    // ── Debug: print what the hardware actually sees ──────────────
                    // inputs196[k] is column-major: k = col*14 + row.
                    // To print row-by-row we must re-index to [row*14+col] for display.
                    printf("\n=== 14x14 input as seen by hardware ===\n");
                    for (int img_row = 0; img_row < 14; img_row++) {
                        for (int img_col = 0; img_col < 14; img_col++) {
                            // k = img_row*14 + img_col (since k/14=img_row, k%14=img_col)
                            uint8_t v = inputs196[img_row * 14 + img_col];
                            const char *shades = " .:-=+*#";
                            int idx = (int)v * 7 / 255;
                            printf("%c%c", shades[idx], shades[idx]);
                        }
                        printf("\n");
                    }
                    printf("=========================================\n\n");

                    // Reset and load inputs to the core
                    dut_reset();
                    for (int i = 0; i < 196; i++) {
                        // Quantize: canvas V∈[0,255] → x_float = V/255 ∈ [0,1]
                        // quantize_input(x, a=-8, b=8, n=8):
                        //   round((x - a) * 255/(b-a)) = round(V/16 + 127.5)
                        float val_f = (float)inputs196[i] / 16.0f + 127.5f;
                        int val_i = (int)std::round(val_f);
                        if (val_i < 0) val_i = 0;
                        if (val_i > 255) val_i = 255;

                        if (cur_sample >= 0 && val_i != tb_inputs[cur_sample][i]) {
                            printf("Mismatch at %d: val_i=%d, tb=%d\n", i, val_i, tb_inputs[cur_sample][i]);
                        }

                        dut->clk = 0;
                        dut->in_mem_we = 1;
                        dut->in_mem_addr = i;
                        dut->in_mem_din = (uint8_t)val_i;
                        dut->eval();
                        dut->clk = 1;
                        dut->eval();
                    }
                    dut->in_mem_we = 0;
                    
                    // Trigger inference
                    dut->clk = 0;
                    dut->start = 1;
                    dut->eval();
                    dut->clk = 1;
                    dut->eval();
                    dut->clk = 0;
                    dut->start = 0;
                    dut->eval();
                    
                    // Run cycle-by-cycle simulation until done
                    int cycles = 0;
                    while (!dut->done && cycles < 300000) {
                        tick();
                        cycles++;
                    }
                    printf("Inference finished in %d clock cycles.\n", cycles);
                    
                    // Read out final scores
                    int max_digit = 0;
                    int max_score = -999999;
                    for (int o = 0; o < 10; o++) {
                        dut->out_mem_addr = o;
                        dut->eval();
                        int score = (uint8_t)dut->out_mem_dout;
                        scores[o] = score;
                        if (score > max_score) {
                            max_score = score;
                            max_digit = o;
                        }
                    }
                    result = max_digit;
                    printf("Result score: ");
                    for (int d = 0; d < 10; d++) {
                        printf("%d:%d ", d, scores[d]);
                    }
                    printf(" -> Predicted: %d\n", result);
                    break;
                }
                case SDLK_c: case SDLK_BACKSPACE:
                    memset(canvas, 0, sizeof(canvas));
                    std::fill(scores.begin(), scores.end(), 0);
                    result = -1;
                    cur_sample = -1;  // back to freehand mode
                    break;
                case SDLK_ESCAPE:
                    running = false;
                    break;
                case SDLK_n: {
                    // Next test sample
                    int next = (cur_sample < 0) ? 0 : (cur_sample + 1) % tb_count;
                    load_sample_to_canvas(next);
                    std::fill(scores.begin(), scores.end(), 0);
                    result = -1;
                    printf("Sample %d  (expected: %d)\n", next, tb_expected_digit[next]);
                    break;
                }
                case SDLK_p: {
                    // Previous test sample
                    int prev = (cur_sample <= 0) ? tb_count - 1 : cur_sample - 1;
                    load_sample_to_canvas(prev);
                    std::fill(scores.begin(), scores.end(), 0);
                    result = -1;
                    printf("Sample %d  (expected: %d)\n", prev, tb_expected_digit[prev]);
                    break;
                }
                case SDLK_EQUALS: case SDLK_PLUS:
                    if (brush < 5) brush++;
                    break;
                case SDLK_MINUS:
                    if (brush > 1) brush--;
                    break;
                }
                break;

            case SDL_MOUSEBUTTONDOWN:
                if (ev.button.x < CSIZE) {
                    mouse_dn = true;
                    erasing  = (ev.button.button == SDL_BUTTON_RIGHT);
                    paint(ev.button.x / CELL, ev.button.y / CELL, erasing);
                    cur_sample = -1;
                } else {
                    // Panel buttons
                    if (ev.button.x >= CLS_BTN.x && ev.button.x < CLS_BTN.x + CLS_BTN.w &&
                        ev.button.y >= CLS_BTN.y && ev.button.y < CLS_BTN.y + CLS_BTN.h) {
                        // Trigger simulation run (same as SPACE / ENTER)
                        SDL_Event push_ev;
                        push_ev.type = SDL_KEYDOWN;
                        push_ev.key.keysym.sym = SDLK_RETURN;
                        SDL_PushEvent(&push_ev);
                    }
                    if (ev.button.x >= CLR_BTN.x && ev.button.x < CLR_BTN.x + CLR_BTN.w &&
                        ev.button.y >= CLR_BTN.y && ev.button.y < CLR_BTN.y + CLR_BTN.h) {
                        memset(canvas, 0, sizeof(canvas));
                        std::fill(scores.begin(), scores.end(), 0);
                        result = -1;
                        cur_sample = -1;
                    }
                }
                break;

            case SDL_MOUSEBUTTONUP:
                mouse_dn = false;
                break;

            case SDL_MOUSEMOTION:
                if (mouse_dn && ev.motion.x >= 0 && ev.motion.x < CSIZE &&
                                ev.motion.y >= 0 && ev.motion.y < WIN_H) {
                    paint(ev.motion.x / CELL, ev.motion.y / CELL, erasing);
                    cur_sample = -1;
                }
                break;
            }
        }

        render(ren, scores, result, false);
        SDL_Delay(16);
    }

    SDL_DestroyRenderer(ren);
    SDL_DestroyWindow(win);
    SDL_Quit();
    
    dut->final();
    delete dut;
    return 0;
}

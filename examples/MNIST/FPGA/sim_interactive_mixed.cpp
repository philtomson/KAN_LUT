// sim_interactive_mixed.cpp - Interactive KAN-LUT unrolled digit classifier simulation
//
// Draw a digit on the 28×28 canvas (left-click drag, right-click erases).
// Click CLASSIFY or press ENTER to run inference through the KAN verilated model.
//
// Keys:  ENTER/SPACE = classify   C/BKSP = clear   ESC = quit
//        +/- = grow/shrink brush

#include <SDL2/SDL.h>
#include <verilated.h>
#include "Vmnist_kan_top.h"

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

// ── Test sample bank (from tb_data_mixed.txt) ──────────────────────────────────────
static uint8_t tb_inputs[100][196];    // 1-bit input (0 or 1) represented as hex
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

// ── Verilator DUT ─────────────────────────────────────────────────────────────
static Vmnist_kan_top *dut;

static void tick() {
    dut->clk = 0;
    dut->eval();
    dut->clk = 1;
    dut->eval();
}

static void dut_reset() {
    dut->rst = 1;
    for (int w = 0; w < 7; w++) {
        dut->in_val[w] = 0;
    }
    // Clock reset
    for (int i = 0; i < 10; i++) {
        dut->clk = 0; dut->eval();
        dut->clk = 1; dut->eval();
    }
    dut->rst = 0;
}

// ── Simple 2×2 average pool, filling dst196 ──────────────────────────────────
static void load_sample_to_canvas(int idx) {
    if (idx < 0 || idx >= tb_count) return;
    cur_sample = idx;
    memset(canvas, 0, sizeof(canvas));
    for (int k = 0; k < 196; k++) {
        int img_col = k % 14;  // x / horizontal
        int img_row = k / 14;  // y / vertical
        uint8_t q   = tb_inputs[idx][k];
        // For mixed precision, the input is 1-bit, represented as 0 or 255.
        uint8_t v = (q > 0) ? 255 : 0;
        for (int dr = 0; dr < 2; dr++)
            for (int dc = 0; dc < 2; dc++)
                canvas[(img_row*2+dr)*GRID + (img_col*2+dc)] = v;
    }
}

static void simple_pool_canvas(const uint8_t *src28, uint8_t *dst196) {
    for (int k = 0; k < 196; k++) {
        int img_col = k % 14;
        int img_row = k / 14;
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
        int exp = tb_expected_digit[cur_sample];
        SDL_SetRenderDrawColor(ren, 80, 60, 120, 255);
        SDL_Rect sbox = {CSIZE+6, 104, PANEL-12, 22};
        SDL_RenderFillRect(ren, &sbox);
        SDL_SetRenderDrawColor(ren, 200, 180, 255, 255);
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

        float fraction = (float)(scores[d] - min_score) / (float)score_range;
        int bar_w = (int)(fraction * (PANEL - 40));
        if (bar_w < 4) bar_w = 4;

        SDL_SetRenderDrawColor(ren, 48, 48, 58, 255);
        SDL_Rect bg_card = {CSIZE+6, y+2, PANEL-12, BAR_H-4};
        SDL_RenderFillRect(ren, &bg_card);

        if (active) {
            SDL_SetRenderDrawColor(ren, 46, 184, 114, 255);
        } else {
            SDL_SetRenderDrawColor(ren, 68, 115, 196, 255);
        }
        SDL_Rect fill_rect = {CSIZE+32, y+10, bar_w, BAR_H-20};
        SDL_RenderFillRect(ren, &fill_rect);

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
    dut = new Vmnist_kan_top;

    // Load test sample bank from tb_data_mixed.txt
    {
        FILE *f2 = fopen("tb_data_mixed.txt", "r");
        if (!f2) f2 = fopen("../examples/MNIST/FPGA/tb_data_mixed.txt", "r");
        if (!f2) f2 = fopen("examples/MNIST/FPGA/tb_data_mixed.txt", "r");
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
                    int val_signed = (int)(uint8_t)v;
                    if (val_signed & 0x20) {
                        val_signed |= ~0x3f;
                    }
                    if (val_signed > ref_max) {
                        ref_max = val_signed;
                        ref_argmax = o;
                    }
                }
                fgetc(f2); // newline
                tb_expected_digit[s] = ref_argmax;
                tb_count++;
            }
            fclose(f2);
            printf("Loaded %d mixed-precision test samples from tb_data_mixed.txt\n", tb_count);
        } else {
            printf("WARNING: tb_data_mixed.txt not found!\n");
        }
    }

    dut_reset();

    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window *win = SDL_CreateWindow(
        "Mixed-Precision KAN unrolled Simulator  |  ENTER=classify  N/P=next/prev sample  C=clear",
        SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, WIN_W, WIN_H, SDL_WINDOW_SHOWN);
    SDL_Renderer *ren = SDL_CreateRenderer(win, -1, SDL_RENDERER_ACCELERATED);

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
                    
                    uint8_t inputs196[196];
                    simple_pool_canvas(canvas, inputs196);

                    printf("\n=== 14x14 input as seen by hardware ===\n");
                    for (int img_row = 0; img_row < 14; img_row++) {
                        for (int img_col = 0; img_col < 14; img_col++) {
                            uint8_t v = inputs196[img_row * 14 + img_col];
                            const char *shades = " .:-=+*#";
                            int idx = (int)v * 7 / 255;
                            printf("%c%c", shades[idx], shades[idx]);
                        }
                        printf("\n");
                    }
                    printf("=========================================\n\n");

                    // Reset and load inputs
                    dut_reset();
                    for (int w = 0; w < 7; w++) {
                        dut->in_val[w] = 0;
                    }
                    for (int i = 0; i < 196; i++) {
                        int word_idx = i / 32;
                        int bit_idx = i % 32;
                        // Threshold inputs196[i] > 0 for 1-bit input activation
                        uint32_t bit = (inputs196[i] > 0) ? 1 : 0;
                        dut->in_val[word_idx] |= (bit << bit_idx);
                    }
                    
                    // Propagate through the unrolled core's 2 pipeline stages
                    for (int cycle = 0; cycle < 2; cycle++) {
                        tick();
                    }

                    // Extract the 60-bit packed output: logic signed [9:0][5:0] out_val
                    uint64_t raw_out = dut->out_val;
                    int max_digit = 0;
                    int max_score = -999999;
                    printf("Result score: ");
                    for (int o = 0; o < 10; o++) {
                        int64_t score_6bit = (raw_out >> (o * 6)) & 0x3F;
                        if (score_6bit & 0x20) {
                            score_6bit |= ~0x3F; // Sign extend 6-bit to 64-bit
                        }
                        int score = (int)score_6bit;
                        scores[o] = score;
                        printf("%d:%d ", o, score);
                        if (score > max_score) {
                            max_score = score;
                            max_digit = o;
                        }
                    }
                    result = max_digit;
                    printf(" -> Predicted: %d\n", result);
                    break;
                }
                case SDLK_c: case SDLK_BACKSPACE:
                    memset(canvas, 0, sizeof(canvas));
                    std::fill(scores.begin(), scores.end(), 0);
                    result = -1;
                    cur_sample = -1;
                    break;
                case SDLK_ESCAPE:
                    running = false;
                    break;
                case SDLK_n: {
                    int next = (cur_sample < 0) ? 0 : (cur_sample + 1) % tb_count;
                    load_sample_to_canvas(next);
                    std::fill(scores.begin(), scores.end(), 0);
                    result = -1;
                    printf("Sample %d  (expected: %d)\n", next, tb_expected_digit[next]);
                    break;
                }
                case SDLK_p: {
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
                    if (ev.button.x >= CLS_BTN.x && ev.button.x < CLS_BTN.x + CLS_BTN.w &&
                        ev.button.y >= CLS_BTN.y && ev.button.y < CLS_BTN.y + CLS_BTN.h) {
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

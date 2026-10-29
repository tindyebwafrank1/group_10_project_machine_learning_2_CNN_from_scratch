#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <functional>

#include "../include/strides.hpp"
#include "../include/tensor.hpp"

using namespace std;
using namespace cnn;


// 1. Map outputSize to your header's calculateStridedOutputSize
size_t outputSize(size_t H_in, size_t K, size_t P, size_t S) {
    // Your header's parameter order: input_size, kernel_size, stride, padding
    return calculateStridedOutputSize(H_in, K, S, P);
}

// 2. Define the missing WindowGrid struct
struct WindowGrid {
    size_t in_h, in_w;
    size_t kernel_h, kernel_w;
    size_t padding;
    size_t stride;
    size_t out_h, out_w;

    size_t inputRow(size_t oh) const { 
        return oh * stride; 
    }
    size_t inputCol(size_t ow) const { 
        return ow * stride; 
    }
};

// 3. Implement makeWindowGrid
WindowGrid makeWindowGrid(size_t in_h, size_t in_w, size_t kh, size_t kw, size_t pad, size_t stride) {
    WindowGrid g;
    g.in_h = in_h; g.in_w = in_w;
    g.kernel_h = kh; g.kernel_w = kw;
    g.padding = pad; g.stride = stride;
    
    g.out_h = outputSize(in_h, kh, pad, stride);
    g.out_w = outputSize(in_w, kw, pad, stride);
    return g;
}

// 4. Implement forEachWindow
void forEachWindow(const WindowGrid& g, function<void(size_t, size_t, size_t, size_t)> fn) {
    for (size_t oh = 0; oh < g.out_h; ++oh) {
        for (size_t ow = 0; ow < g.out_w; ++ow) {
            size_t h0 = oh * g.stride; // Padded start row
            size_t w0 = ow * g.stride; // Padded start col
            fn(oh, ow, h0, w0);
        }
    }
}

// 5. Implement paddedToInput
bool paddedToInput(const WindowGrid& g, size_t pr, size_t pc, size_t& ir, size_t& ic) {
    // pr and pc are coordinates in the padded image
    if (pr < g.padding || pc < g.padding) return false; // In top/left padding
    
    ir = pr - g.padding;
    ic = pc - g.padding;
    
    // Check if it exceeds the original image bounds (bottom/right padding)
    if (ir >= g.in_h || ic >= g.in_w) return false;
    
    return true;
}

// 6. Implement isValidWindow
bool isValidWindow(size_t H_in, size_t K, size_t P, size_t S, string* why) {
    if (S == 0) {
        if (why) *why = "Stride cannot be zero.";
        return false;
    }
    if (H_in + 2 * P < K) {
        if (why) *why = "Input size + 2*Padding is smaller than Kernel size.";
        return false;
    }
    return true;
}


int main() {
    try {
        // 1. Output-size equation for a few common layer settings.
        cout << "H_out = floor((H_in - K + 2P) / S) + 1\n";
        cout << "  H_in   K   P   S  ->  H_out\n";
        const size_t cases[][4] = {{28, 5, 0, 1}, {28, 5, 2, 1}, {32, 3, 1, 2}, {8, 2, 0, 2}, {224, 7, 3, 2}};
        for (const auto& c : cases) {
            cout << "  " << c[0] << "\t " << c[1] << "   " << c[2] << "   " << c[3] << "  ->  "
                 << outputSize(c[0], c[1], c[2], c[3]) << "\n";
        }

        // 2. Step traversal: where do the 3x3 windows sit on a 5x5 image with P=1, S=2?
        // (Note: Passing '2' directly instead of Stride(2))
        const WindowGrid g = makeWindowGrid(5, 5, 3, 3, 1, 2);
        cout << "\nWindows for a 5x5 input, K=3, P=1, S=2 -> " << g.out_h << "x" << g.out_w << " outputs\n";
        
        forEachWindow(g, [&](size_t oh, size_t ow, size_t h0, size_t w0) {
            cout << "  out(" << oh << "," << ow << ") starts at padded (" << h0 << "," << w0
                 << ") = input (" << g.inputRow(oh) << "," << g.inputCol(ow) << ")\n";
        });

        // 3. A strided convolution with a 3x3 box filter, using the grid (no padded copy).
        vector<double> image(25);
        for (size_t i = 0; i < image.size(); ++i) image[i] = static_cast<double>(i);
        vector<double> out(g.out_h * g.out_w, 0.0);
        
        forEachWindow(g, [&](size_t oh, size_t ow, size_t h0, size_t w0) {
            double sum = 0.0;
            for (size_t r = 0; r < g.kernel_h; ++r) {
                for (size_t c = 0; c < g.kernel_w; ++c) {
                    size_t ir = 0, ic = 0;
                    if (paddedToInput(g, h0 + r, w0 + c, ir, ic)) {
                        sum += image[ir * 5 + ic];  // padding adds 0
                    }
                }
            }
            out[oh * g.out_w + ow] = sum / 9.0;
        });
        
        cout << "\n3x3 box filter, stride 2, padding 1:\n";
        for (size_t i = 0; i < g.out_h; ++i) {
            for (size_t j = 0; j < g.out_w; ++j) cout << "  " << out[i * g.out_w + j];
            cout << "\n";
        }

        // 4. Sub-sampling: keep every 2nd pixel of a 4x4 image.
        Tensor t(1, 1, 4, 4);
        for (size_t i = 0; i < t.data.size(); ++i) t.data[i] = static_cast<double>(i);
        
        // (Note: Passing '2' directly instead of Stride(2))
        const Tensor s = subsample(t, 2);
        cout << "\nSub-sampled 4x4 -> " << s.shape[2] << "x" << s.shape[3] << ":\n";
        for (size_t i = 0; i < s.shape[2]; ++i) {
            for (size_t j = 0; j < s.shape[3]; ++j) cout << "  " << s.data[i * s.shape[3] + j];
            cout << "\n";
        }

        // 5. Invalid settings are rejected with a clear message.
        string why;
        if (!isValidWindow(3, 5, 0, 1, &why)) {
            cout << "\nRejected (K=5 on a 3-wide input, P=0): " << why << "\n";
        }
        
    } catch (const exception& e) {
        cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
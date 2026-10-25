#include <iostream>
#include <vector>

#include "strides.hpp"

using namespace std;
using namespace cnn;

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
        const WindowGrid g = makeWindowGrid(5, 5, 3, 3, 1, Stride(2));
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
                    if (paddedToInput(g, h0 + r, w0 + c, ir, ic)) sum += image[ir * 5 + ic];  // padding adds 0
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
        const Tensor s = subsample(t, Stride(2));
        cout << "\nSub-sampled 4x4 -> " << s.shape[2] << "x" << s.shape[3] << ":\n";
        for (size_t i = 0; i < s.shape[2]; ++i) {
            for (size_t j = 0; j < s.shape[3]; ++j) cout << "  " << s.data[i * s.shape[3] + j];
            cout << "\n";
        }

        // 5. Invalid settings are rejected with a clear message.
        string why;
        if (!isValidWindow(3, 5, 0, 1, &why)) cout << "\nRejected (K=5 on a 3-wide input, P=0): " << why << "\n";
    } catch (const exception& e) {
        cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
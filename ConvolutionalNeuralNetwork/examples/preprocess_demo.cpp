#include <iostream>
#include <vector>

#include "cnn/preprocess.hpp"

using namespace std;
using namespace cnn;

int main() {
    try {
        // 4x4 RGB gradient as interleaved bytes, the same layout a file loader produces.
        // With Frank's loader, replace this block with:
        //   Tensor raw = fromInterleaved(img.pixels.data(), img.pixels.size(),
        //                                img.height, img.width, img.channels);
        vector<unsigned char> px(4 * 4 * 3);
        for (size_t y = 0; y < 4; ++y) {
            for (size_t x = 0; x < 4; ++x) {
                px[(y * 4 + x) * 3 + 0] = static_cast<unsigned char>(x * 85);
                px[(y * 4 + x) * 3 + 1] = static_cast<unsigned char>(y * 85);
                px[(y * 4 + x) * 3 + 2] = 128;
            }
        }
        const Tensor raw = fromInterleaved(px.data(), px.size(), 4, 4, 3);

        PreprocessConfig cfg;
        cfg.target_channels = 1;  // RGB -> grayscale
        cfg.target_height = 2;    // 4x4 -> 2x2
        cfg.target_width = 2;
        const Tensor out = preprocess(raw, cfg);

        cout << "input  shape: {" << raw.shape[0] << "," << raw.shape[1] << ","
             << raw.shape[2] << "," << raw.shape[3] << "}\n";
        cout << "output shape: {" << out.shape[0] << "," << out.shape[1] << ","
             << out.shape[2] << "," << out.shape[3] << "}\n";
        for (size_t y = 0; y < out.shape[2]; ++y) {
            for (size_t x = 0; x < out.shape[3]; ++x) {
                cout << out.data[y * out.shape[3] + x] << ' ';
            }
            cout << '\n';
        }
    } catch (const exception& e) {
        cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
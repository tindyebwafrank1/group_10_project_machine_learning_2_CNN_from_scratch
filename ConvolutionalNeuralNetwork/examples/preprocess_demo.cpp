

#include <iostream>
#include <vector>
#include <exception>

#include "../include/preprocess.hpp"

using namespace std;
using namespace cnn;

int main()
{
    try
    {
        // Create a 4x4 RGB test image.
        vector<unsigned char> px(4 * 4 * 3);

        for (size_t y = 0; y < 4; ++y)
        {
            for (size_t x = 0; x < 4; ++x)
            {
                px[(y * 4 + x) * 3 + 0] =
                    static_cast<unsigned char>(x * 85);

                px[(y * 4 + x) * 3 + 1] =
                    static_cast<unsigned char>(y * 85);

                px[(y * 4 + x) * 3 + 2] = 128;
            }
        }

        const Tensor raw =
            fromInterleaved(px.data(), px.size(), 4, 4, 3);

        // Convert RGB to grayscale and resize from 4x4 to 2x2.
        PreprocessConfig cfg;
        cfg.target_channels = 1;
        cfg.target_height = 2;
        cfg.target_width = 2;

        const Tensor out = preprocess(raw, cfg);

        cout << "Input shape: {"
             << raw.shape[0] << ","
             << raw.shape[1] << ","
             << raw.shape[2] << ","
             << raw.shape[3] << "}\n";

        cout << "Output shape: {"
             << out.shape[0] << ","
             << out.shape[1] << ","
             << out.shape[2] << ","
             << out.shape[3] << "}\n";

        cout << "Preprocessed values:\n";

        for (size_t y = 0; y < out.shape[2]; ++y)
        {
            for (size_t x = 0; x < out.shape[3]; ++x)
            {
                cout << out.data[y * out.shape[3] + x] << ' ';
            }

            cout << '\n';
        }
    }
    catch (const exception& e)
    {
        cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}

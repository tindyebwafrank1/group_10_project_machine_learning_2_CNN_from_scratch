// Pipes a synthetic 28x28 grayscale image through a 3-block feature extractor.
// Real pipeline: the input Tensor comes from the image loader/preprocessor (M1/M2/M3).
#include <iostream>
#include "cnn_library/feature_block.hpp"

int main() {
    using namespace CNN;

    Tensor image(1, 1, 28, 28);
    for (size_t h = 0; h < 28; ++h)
        for (size_t w = 0; w < 28; ++w)
            image(0, 0, h, w) = double(h + w) / 54.0; // simple gradient in [0, 1]

    FeatureExtractor fx(1);
    fx.add_block(4, 3, 1, 1)                      // 28x28 -> conv(same) 28x28 -> pool 14x14
      .add_block(8, 3, 1, 1)                      // 14x14 -> 7x7
      .add_block(16, 3, 1, 0, Activation::Tanh);  // 7x7 -> conv 5x5 -> pool 2x2

    auto expected = fx.output_shape(28, 28);
    Tensor features = fx.forward(image);

    std::cout << "Input:    1 x 1 x 28 x 28\n";
    std::cout << "Expected: " << expected[0] << " x " << expected[1] << " x " << expected[2] << "\n";
    std::cout << "Output:   " << features.shape[0] << " x " << features.shape[1] << " x "
              << features.shape[2] << " x " << features.shape[3] << "\n";
    std::cout << "Feature values (channel 0): ";
    for (size_t h = 0; h < features.shape[2]; ++h)
        for (size_t w = 0; w < features.shape[3]; ++w) std::cout << features(0, 0, h, w) << " ";
    std::cout << "\n";
    return 0;
}

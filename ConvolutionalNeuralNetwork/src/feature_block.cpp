
#include "feature_block.hpp"
#include "multi_channel_convolution.hpp"

#include <algorithm>
#include <stdexcept>
#include <cstddef>

using namespace std;

namespace cnn {

FeatureBlock::FeatureBlock(
    const Tensor& convolution_filters,
    size_t pooling_window,
    size_t pooling_stride
)
    : filters(convolution_filters),
      pool_window(pooling_window),
      pool_stride(pooling_stride) {

    if (pool_window == 0 || pool_stride == 0) {
        throw invalid_argument(
            "Pooling window and stride must be positive"
        );
    }

    if (filters.shape.size() != 4) {
        throw invalid_argument(
            "Convolution filters must be a 4-D tensor"
        );
    }
}

Tensor FeatureBlock::forward(const Tensor& input) const {

    // Step 1: Multi-channel, multi-filter convolution.
    Tensor features = multiChannelConvolution(input, filters);

    // Step 2: Apply ReLU activation element by element.
    for (size_t i = 0; i < features.data.size(); ++i) {
        features.data[i] = max(0.0, features.data[i]);
    }

    // Step 3: Max pooling.
    const size_t batch_size = features.shape[0];
    const size_t channels = features.shape[1];
    const size_t height = features.shape[2];
    const size_t width = features.shape[3];

    if (height < pool_window || width < pool_window) {
        throw invalid_argument(
            "Pooling window cannot exceed feature dimensions"
        );
    }

    // Pooling uses no padding.
    const size_t output_height =
        (height - pool_window) / pool_stride + 1;

    const size_t output_width =
        (width - pool_window) / pool_stride + 1;

    Tensor pooled(
        batch_size,
        channels,
        output_height,
        output_width
    );

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < channels; ++c) {
            for (size_t h = 0; h < output_height; ++h) {
                for (size_t w = 0; w < output_width; ++w) {

                    double maximum =
                        features(
                            n, c,
                            h * pool_stride,
                            w * pool_stride
                        );

                    for (size_t r = 0; r < pool_window; ++r) {
                        for (size_t s = 0; s < pool_window; ++s) {

                            const double value = features(
                                n, c,
                                h * pool_stride + r,
                                w * pool_stride + s
                            );

                            maximum = max(maximum, value);
                        }
                    }

                    pooled(n, c, h, w) = maximum;
                }
            }
        }
    }

    return pooled;
}

} 

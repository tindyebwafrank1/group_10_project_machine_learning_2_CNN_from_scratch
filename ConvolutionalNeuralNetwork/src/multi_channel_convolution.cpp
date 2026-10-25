
#include "multi_channel_convolution.hpp"

#include <stdexcept>
#include <cstddef>

using namespace std;

namespace cnn {

Tensor multiChannelConvolution(
    const Tensor& input,
    const Tensor& filters
) {
    // Check that input and filters are 4-D tensors.
    if (input.shape.size() != 4 ||
        filters.shape.size() != 4) {
        throw invalid_argument(
            "Input and filters must be 4-D tensors"
        );
    }

    const size_t batch_size = input.shape[0];
    const size_t input_channels = input.shape[1];
    const size_t input_height = input.shape[2];
    const size_t input_width = input.shape[3];

    const size_t output_channels = filters.shape[0];
    const size_t filter_channels = filters.shape[1];
    const size_t kernel_height = filters.shape[2];
    const size_t kernel_width = filters.shape[3];

    // Input channels must match filter channels.
    if (input_channels != filter_channels) {
        throw invalid_argument(
            "Input channels must match filter channels"
        );
    }

    // Validate kernel dimensions.
    if (kernel_height == 0 || kernel_width == 0) {
        throw invalid_argument(
            "Kernel dimensions must be positive"
        );
    }

    if (input_height < kernel_height ||
        input_width < kernel_width) {
        throw invalid_argument(
            "Kernel cannot be larger than input"
        );
    }

    if (output_channels == 0) {
        throw invalid_argument(
            "At least one filter is required"
        );
    }

    // Valid convolution: stride 1, no padding.
    const size_t output_height =
        input_height - kernel_height + 1;

    const size_t output_width =
        input_width - kernel_width + 1;

    // Allocate the output Tensor.
    Tensor output(
        batch_size,
        output_channels,
        output_height,
        output_width
    );

    // Loop through each image in the batch.
    for (size_t n = 0; n < batch_size; ++n) {

        // Loop through each filter.
        for (size_t k = 0; k < output_channels; ++k) {

            // Slide the filter vertically.
            for (size_t h = 0; h < output_height; ++h) {

                // Slide the filter horizontally.
                for (size_t w = 0; w < output_width; ++w) {

                    double sum = 0.0;

                    // Accumulate contributions from every input channel.
                    for (size_t c = 0;
                         c < input_channels; ++c) {

                        for (size_t r = 0;
                             r < kernel_height; ++r) {

                            for (size_t s = 0;
                                 s < kernel_width; ++s) {

                                sum +=
                                    input(n, c, h + r, w + s)
                                    * filters(k, c, r, s);
                            }
                        }
                    }

                    // Save the value in the output feature map.
                    output(n, k, h, w) = sum;
                }
            }
        }
    }

    return output;
}

} // namespace cnn

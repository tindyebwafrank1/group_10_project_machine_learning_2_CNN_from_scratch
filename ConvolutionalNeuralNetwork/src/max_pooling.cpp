
#include "max_pooling.hpp"

#include <stdexcept>
#include <limits>

using namespace std;

MaxPoolingResult maxPool2D(
    const Tensor& input,
    const PoolingParams& params)
{
    // Confirm that the input is a 4D tensor: N, C, H, W.
    if (input.shape.size() != 4)
    {
        throw invalid_argument(
            "Max pooling requires a 4D NCHW tensor."
        );
    }

    if (params.window_h == 0 || params.window_w == 0 ||
        params.stride_h == 0 || params.stride_w == 0)
    {
        throw invalid_argument(
            "Pooling window and stride values must be positive."
        );
    }

    size_t N = input.shape[0];
    size_t C = input.shape[1];
    size_t H = input.shape[2];
    size_t W = input.shape[3];

    if (params.window_h > H || params.window_w > W)
    {
        throw invalid_argument(
            "Pooling window cannot be larger than the input."
        );
    }

    // Verify that the tensor data matches its declared shape.
    if (N == 0 || C == 0 || H == 0 || W == 0 ||
        C > numeric_limits<size_t>::max() / N ||
        H > numeric_limits<size_t>::max() / (N * C) ||
        W > numeric_limits<size_t>::max() / (N * C * H) ||
        input.data.size() != N * C * H * W)
    {
        throw invalid_argument("Invalid input tensor dimensions or data.");
    }

    // Calculate the dimensions of the pooled output.
    size_t output_h =
        1 + (H - params.window_h) / params.stride_h;

    size_t output_w =
        1 + (W - params.window_w) / params.stride_w;

    Tensor output(N, C, output_h, output_w);

    vector<size_t> max_indices(output.size());

    // Process every batch and channel independently.
    for (size_t n = 0; n < N; n++)
    {
        for (size_t c = 0; c < C; c++)
        {
            for (size_t oh = 0; oh < output_h; oh++)
            {
                for (size_t ow = 0; ow < output_w; ow++)
                {
                    size_t start_h = oh * params.stride_h;
                    size_t start_w = ow * params.stride_w;

                    // Initialize from the first element in the window.
                    size_t max_h = start_h;
                    size_t max_w = start_w;
                    double max_value = input(n, c, max_h, max_w);

                    // Search the pooling window for its maximum.
                    for (size_t kh = 0; kh < params.window_h; kh++)
                    {
                        for (size_t kw = 0; kw < params.window_w; kw++)
                        {
                            size_t h = start_h + kh;
                            size_t w = start_w + kw;

                            double value = input(n, c, h, w);

                            // Keep the first maximum if values are tied.
                            if (value > max_value)
                            {
                                max_value = value;
                                max_h = h;
                                max_w = w;
                            }
                        }
                    }

                    // Store the maximum in the output tensor.
                    output(n, c, oh, ow) = max_value;

                    // Save the corresponding flat input index.
                    size_t output_index =
                        ((n * C + c) * output_h + oh) * output_w + ow;

                    size_t input_index =
                        ((n * C + c) * H + max_h) * W + max_w;

                    max_indices[output_index] = input_index;
                }
            }
        }
    }

    return {output, max_indices};
}

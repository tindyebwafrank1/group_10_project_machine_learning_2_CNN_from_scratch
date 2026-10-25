
#include "pooling_backward.hpp"

#include <limits>
#include <stdexcept>

using namespace std;

// Checks that a tensor has the expected NCHW dimensions.
static void validateTensorShape(const Tensor& tensor)
{
    if (tensor.shape.size() != 4)
    {
        throw invalid_argument("Expected a 4D NCHW tensor.");
    }

    size_t N = tensor.shape[0];
    size_t C = tensor.shape[1];
    size_t H = tensor.shape[2];
    size_t W = tensor.shape[3];

    if (N == 0 || C == 0 || H == 0 || W == 0 ||
        C > numeric_limits<size_t>::max() / N ||
        H > numeric_limits<size_t>::max() / (N * C) ||
        W > numeric_limits<size_t>::max() / (N * C * H) ||
        tensor.data.size() != N * C * H * W)
    {
        throw invalid_argument("Invalid tensor dimensions or data.");
    }
}

// Checks that the upstream gradient matches the forward output.
static void validateGradientShape(
    const Tensor& output,
    const Tensor& upstream_gradient)
{
    validateTensorShape(output);
    validateTensorShape(upstream_gradient);

    if (output.shape != upstream_gradient.shape)
    {
        throw invalid_argument(
            "Upstream gradient shape must match the pooling output shape."
        );
    }
}

Tensor maxPool2DBackward(
    const Tensor& input,
    const MaxPoolingResult& forward_result,
    const Tensor& upstream_gradient)
{
    validateTensorShape(input);

    validateGradientShape(
        forward_result.output,
        upstream_gradient
    );

    if (forward_result.max_indices.size() !=
        upstream_gradient.data.size())
    {
        throw invalid_argument(
            "Saved maximum indices must match the output size."
        );
    }

    Tensor input_gradient(
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3]
    );

    input_gradient.fill(0.0);

    // Route each output gradient to its saved maximum location.
    for (size_t i = 0; i < upstream_gradient.data.size(); i++)
    {
        size_t input_index = forward_result.max_indices[i];

        if (input_index >= input_gradient.data.size())
        {
            throw invalid_argument(
                "A saved maximum index is outside the input tensor."
            );
        }

        // += also handles repeated indices safely.
        input_gradient.data[input_index] +=
            upstream_gradient.data[i];
    }

    return input_gradient;
}

Tensor averagePool2DBackward(
    const Tensor& input,
    const PoolingParams& params,
    const Tensor& upstream_gradient)
{
    validateTensorShape(input);

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

    size_t output_h =
        1 + (H - params.window_h) / params.stride_h;

    size_t output_w =
        1 + (W - params.window_w) / params.stride_w;

    // Verify the gradient shape expected from valid average pooling.
    if (upstream_gradient.shape.size() != 4 ||
        upstream_gradient.shape[0] != N ||
        upstream_gradient.shape[1] != C ||
        upstream_gradient.shape[2] != output_h ||
        upstream_gradient.shape[3] != output_w ||
        upstream_gradient.data.size() != N * C * output_h * output_w)
    {
        throw invalid_argument(
            "Upstream gradient shape does not match average pooling output."
        );
    }

    Tensor input_gradient(N, C, H, W);
    input_gradient.fill(0.0);

    double divisor =
        static_cast<double>(params.window_h) *
        static_cast<double>(params.window_w);

    // Each pooled gradient is shared equally across its window.
    for (size_t n = 0; n < N; n++)
    {
        for (size_t c = 0; c < C; c++)
        {
            for (size_t oh = 0; oh < output_h; oh++)
            {
                for (size_t ow = 0; ow < output_w; ow++)
                {
                    double contribution =
                        upstream_gradient(n, c, oh, ow) / divisor;

                    size_t start_h = oh * params.stride_h;
                    size_t start_w = ow * params.stride_w;

                    for (size_t kh = 0; kh < params.window_h; kh++)
                    {
                        for (size_t kw = 0; kw < params.window_w; kw++)
                        {
                            size_t h = start_h + kh;
                            size_t w = start_w + kw;

                            // Overlapping windows accumulate gradients.
                            input_gradient(n, c, h, w) += contribution;
                        }
                    }
                }
            }
        }
    }

    return input_gradient;
}

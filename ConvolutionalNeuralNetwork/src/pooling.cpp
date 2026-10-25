
#include "pooling.hpp"
#include "tensor.hpp"

#include <stdexcept>

using namespace std;

namespace cnn {

// Initialize the base pooling layer.
Pooling2D::Pooling2D(PoolingParams params)
    : params_(params)
{
    params_.validate();
}

// Initialize the maximum pooling layer.
MaxPooling2D::MaxPooling2D(PoolingParams params)
    : Pooling2D(params)
{
}

// Perform maximum pooling on a 4D tensor (N, C, H, W).
Tensor MaxPooling2D::forward(const Tensor& input) const
{
    if (input.shape.size() != 4)
    {
        throw invalid_argument(
            "MaxPooling2D: input must be a 4-D tensor"
        );
    }

    params_.validate();

    size_t batch_size = input.shape[0];
    size_t channels = input.shape[1];
    size_t input_h = input.shape[2];
    size_t input_w = input.shape[3];

    size_t output_h = output_height(input_h);
    size_t output_w = output_width(input_w);

    Tensor output(batch_size, channels, output_h, output_w);

    for (size_t n = 0; n < batch_size; ++n)
    {
        for (size_t c = 0; c < channels; ++c)
        {
            for (size_t oh = 0; oh < output_h; ++oh)
            {
                for (size_t ow = 0; ow < output_w; ++ow)
                {
                    size_t start_h = oh * params_.stride_h;
                    size_t start_w = ow * params_.stride_w;

                    double maximum = input(
                        n, c, start_h, start_w
                    );

                    for (size_t kh = 0;
                         kh < params_.window_h; ++kh)
                    {
                        for (size_t kw = 0;
                             kw < params_.window_w; ++kw)
                        {
                            double value = input(
                                n,
                                c,
                                start_h + kh,
                                start_w + kw
                            );

                            if (value > maximum)
                            {
                                maximum = value;
                            }
                        }
                    }

                    output(n, c, oh, ow) = maximum;
                }
            }
        }
    }

    return output;
}

// Initialize the average pooling layer.
AveragePooling2D::AveragePooling2D(PoolingParams params)
    : Pooling2D(params)
{
}

// Perform average pooling on a 4D tensor (N, C, H, W).
Tensor AveragePooling2D::forward(const Tensor& input) const
{
    if (input.shape.size() != 4)
    {
        throw invalid_argument(
            "AveragePooling2D: input must be a 4-D tensor"
        );
    }

    params_.validate();

    size_t batch_size = input.shape[0];
    size_t channels = input.shape[1];
    size_t input_h = input.shape[2];
    size_t input_w = input.shape[3];

    size_t output_h = output_height(input_h);
    size_t output_w = output_width(input_w);

    Tensor output(batch_size, channels, output_h, output_w);

    double window_area =
        static_cast<double>(params_.window_h)
        * static_cast<double>(params_.window_w);

    for (size_t n = 0; n < batch_size; ++n)
    {
        for (size_t c = 0; c < channels; ++c)
        {
            for (size_t oh = 0; oh < output_h; ++oh)
            {
                for (size_t ow = 0; ow < output_w; ++ow)
                {
                    double sum = 0.0;

                    size_t start_h = oh * params_.stride_h;
                    size_t start_w = ow * params_.stride_w;

                    for (size_t kh = 0;
                         kh < params_.window_h; ++kh)
                    {
                        for (size_t kw = 0;
                             kw < params_.window_w; ++kw)
                        {
                            sum += input(
                                n,
                                c,
                                start_h + kh,
                                start_w + kw
                            );
                        }
                    }

                    output(n, c, oh, ow) = sum / window_area;
                }
            }
        }
    }

    return output;
}

} // namespace cnn


#include "convolution_backward.hpp"
#include <stdexcept>

using namespace std;

namespace cnn {

// Validate the input, kernels, and upstream gradient shapes.
static void validateBackwardInputs(
    const ::Tensor& input,
    const ::Tensor& kernels,
    const ::Tensor& grad_output
) {
    if (input.shape.size() != 4 ||
        kernels.shape.size() != 4 ||
        grad_output.shape.size() != 4) {
        throw invalid_argument(
            "Convolution backward: all tensors must be 4-D"
        );
    }

    size_t batch_size = input.shape[0];
    size_t input_channels = input.shape[1];
    size_t input_h = input.shape[2];
    size_t input_w = input.shape[3];

    size_t output_channels = kernels.shape[0];
    size_t kernel_channels = kernels.shape[1];
    size_t kernel_h = kernels.shape[2];
    size_t kernel_w = kernels.shape[3];

    if (batch_size == 0 || input_channels == 0 ||
        input_h == 0 || input_w == 0 ||
        output_channels == 0 || kernel_channels == 0 ||
        kernel_h == 0 || kernel_w == 0) {
        throw invalid_argument(
            "Convolution backward: tensor dimensions must be positive"
        );
    }

    if (input_channels != kernel_channels) {
        throw invalid_argument(
            "Convolution backward: input and kernel channels do not match"
        );
    }

    if (kernel_h > input_h || kernel_w > input_w) {
        throw invalid_argument(
            "Convolution backward: kernel is larger than input"
        );
    }

    size_t expected_h = input_h - kernel_h + 1;
    size_t expected_w = input_w - kernel_w + 1;

    if (grad_output.shape[0] != batch_size ||
        grad_output.shape[1] != output_channels ||
        grad_output.shape[2] != expected_h ||
        grad_output.shape[3] != expected_w) {
        throw invalid_argument(
            "Convolution backward: grad_output has incorrect shape"
        );
    }
}

// Calculate the gradient of the loss with respect to the kernels.
// Input shape:   (N, Cin, H, W)
// Kernel shape:  (Cout, Cin, Kh, Kw)
// Gradient shape:(N, Cout, Hout, Wout)
::Tensor convolutionKernelGradient(
    const ::Tensor& input,
    const ::Tensor& kernels,
    const ::Tensor& grad_output
) {
    validateBackwardInputs(input, kernels, grad_output);

    size_t batch_size = input.shape[0];
    size_t input_channels = input.shape[1];
    size_t output_channels = kernels.shape[0];
    size_t kernel_h = kernels.shape[2];
    size_t kernel_w = kernels.shape[3];
    size_t output_h = grad_output.shape[2];
    size_t output_w = grad_output.shape[3];

    ::Tensor grad_kernels(
        output_channels,
        input_channels,
        kernel_h,
        kernel_w
    );

    grad_kernels.fill(0.0);

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t o = 0; o < output_channels; ++o) {
            for (size_t c = 0; c < input_channels; ++c) {
                for (size_t h = 0; h < output_h; ++h) {
                    for (size_t w = 0; w < output_w; ++w) {
                        double upstream = grad_output(n, o, h, w);

                        for (size_t i = 0; i < kernel_h; ++i) {
                            for (size_t j = 0; j < kernel_w; ++j) {
                                grad_kernels(o, c, i, j) +=
                                    input(n, c, h + i, w + j) * upstream;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_kernels;
}

// Calculate the gradient of the loss with respect to the input.
::Tensor convolutionInputGradient(
    const ::Tensor& input,
    const ::Tensor& kernels,
    const ::Tensor& grad_output
) {
    validateBackwardInputs(input, kernels, grad_output);

    size_t batch_size = input.shape[0];
    size_t input_channels = input.shape[1];
    size_t input_h = input.shape[2];
    size_t input_w = input.shape[3];

    size_t output_channels = kernels.shape[0];
    size_t kernel_h = kernels.shape[2];
    size_t kernel_w = kernels.shape[3];

    size_t output_h = grad_output.shape[2];
    size_t output_w = grad_output.shape[3];

    ::Tensor grad_input(
        batch_size,
        input_channels,
        input_h,
        input_w
    );

    grad_input.fill(0.0);

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t o = 0; o < output_channels; ++o) {
            for (size_t h = 0; h < output_h; ++h) {
                for (size_t w = 0; w < output_w; ++w) {
                    double upstream = grad_output(n, o, h, w);

                    for (size_t c = 0; c < input_channels; ++c) {
                        for (size_t i = 0; i < kernel_h; ++i) {
                            for (size_t j = 0; j < kernel_w; ++j) {
                                grad_input(n, c, h + i, w + j) +=
                                    kernels(o, c, i, j) * upstream;
                            }
                        }
                    }
                }
            }
        }
    }

    return grad_input;
}

} // namespace cnn

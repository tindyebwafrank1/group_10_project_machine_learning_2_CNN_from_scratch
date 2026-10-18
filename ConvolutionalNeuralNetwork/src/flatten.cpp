
#include "flatten.hpp"

#include <stdexcept>
#include <limits>

using namespace std;

namespace cnn {

vector<vector<double>> flatten(const ::Tensor& input) {
    // A valid CNN tensor must have four dimensions.
    if (input.shape.size() != 4) {
        throw invalid_argument(
            "Flatten requires a 4D tensor (N, C, H, W)"
        );
    }

    const size_t N = input.shape[0];
    const size_t C = input.shape[1];
    const size_t H = input.shape[2];
    const size_t W = input.shape[3];

    // Check for overflow when calculating the feature count.
    const size_t maxSize = numeric_limits<size_t>::max();

    if (H != 0 && C > maxSize / H) {
        throw overflow_error("Tensor dimensions overflow");
    }

    const size_t CH = C * H;

    if (W != 0 && CH > maxSize / W) {
        throw overflow_error("Tensor dimensions overflow");
    }

    const size_t features = CH * W;

    if (features != 0 && N > maxSize / features) {
        throw overflow_error("Tensor dimensions overflow");
    }

    // Verify that the data size matches the tensor shape.
    if (input.data.size() != N * features) {
        throw invalid_argument(
            "Tensor data size does not match its shape"
        );
    }

    // Create the output matrix: N rows, C*H*W columns.
    vector<vector<double>> output(
        N, vector<double>(features)
    );

    // Copy each tensor element into its flattened position.
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t h = 0; h < H; ++h) {
                for (size_t w = 0; w < W; ++w) {
                    const size_t index =
                        c * (H * W) + h * W + w;

                    output[n][index] = input(n, c, h, w);
                }
            }
        }
    }

    return output;
}

} // namespace cnn

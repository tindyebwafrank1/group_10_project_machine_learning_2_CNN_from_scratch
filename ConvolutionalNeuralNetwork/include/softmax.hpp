#ifndef SOFTMAX_HPP
#define SOFTMAX_HPP

#include <cmath>
#include <cstddef>
#include <vector>
#include <stdexcept>
#include <algorithm>
using namespace std;

namespace cnn {

// Softmax layer (Week 3)
// Input: scores of shape {N, C, 1, 1} (output of the fully connected layer)
// Output: probabilities of the same shape, each sample sums to 1
// forward() is a template so it works with any tensor that has
// .data and .shape (the project Tensor and the FullyConnected Tensor)
class Softmax {
private:
    vector<double> last_output_; // saved probabilities for Week 4

public:
    template <class T>
    T forward(const T& input)
    {
        if (input.shape.size() != 4 || input.shape[2] != 1 || input.shape[3] != 1)
        {
            throw invalid_argument("Softmax: input shape must be {N, C, 1, 1}");
        }
        size_t N = input.shape[0];
        size_t C = input.shape[1];
        if (C == 0 || input.data.size() != N * C)
        {
            throw invalid_argument("Softmax: bad number of classes or data size");
        }
        for (double v : input.data)
        {
            if (!isfinite(v))
            {
                throw invalid_argument("Softmax: input contains NaN or Inf");
            }
        }

        T out = input;
        for (size_t n = 0; n < N; ++n)
        {
            size_t base = n * C;

            // 1. largest score of this sample
            double mx = input.data[base];
            for (size_t c = 1; c < C; ++c)
            {
                mx = max(mx, input.data[base + c]);
            }

            // 2. exp(x - max), keep a running sum
            double sum = 0.0;
            for (size_t c = 0; c < C; ++c)
            {
                double e = exp(input.data[base + c] - mx);
                out.data[base + c] = e;
                sum += e;
            }

            // 3. divide by the sum
            for (size_t c = 0; c < C; ++c)
            {
                out.data[base + c] /= sum;
            }
        }

        last_output_ = out.data;
        return out;
    }

    // probabilities from the last forward() call
    const vector<double>& output() const { return last_output_; }
};

} // namespace cnn

#endif // SOFTMAX_HPP

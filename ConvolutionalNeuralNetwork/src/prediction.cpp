#include "prediction.hpp"
#include <cmath>
#include <stdexcept>

namespace cnn {

std::size_t argmax(const std::vector<double>& scores) {
    if (scores.empty())
        throw std::invalid_argument("argmax: scores is empty");

    std::size_t best = 0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        if (std::isnan(scores[i]))
            throw std::invalid_argument("argmax: scores contains NaN");
        if (scores[i] > scores[best])   // strict '>' keeps the first index on ties
            best = i;
    }
    return best;
}

std::vector<std::size_t> predict(const Tensor& scores) {
    if (scores.shape.size() != 4 || scores.shape[0] == 0 || scores.data.empty())
        throw std::invalid_argument("predict: expected a non-empty (N,...) tensor");

    const std::size_t n = scores.shape[0];
    const std::size_t k = scores.data.size() / n;   // scores per sample
    if (k == 0 || k * n != scores.data.size())
        throw std::invalid_argument("predict: data size not divisible by N");

    std::vector<std::size_t> labels;
    labels.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        std::vector<double> row(scores.data.begin() + i * k,
                                scores.data.begin() + (i + 1) * k);
        labels.push_back(argmax(row));
    }
    return labels;
}

}  // namespace cnn
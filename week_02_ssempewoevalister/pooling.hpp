#ifndef POOLING_HPP
#define POOLING_HPP
// STAND-IN for Mugarura's max pooling and Joan's average pooling.
#include <algorithm>
#include <limits>
#include <stdexcept>
#include "cnn_library/tensor.hpp"
namespace CNN {
    class Pooling {
    public:
        enum class Mode { Max, Average };
    private:
        size_t win, str; Mode mode_;
    public:
        Pooling(size_t window = 2, size_t stride = 2, Mode m = Mode::Max) : win(window), str(stride), mode_(m) {
            if (win == 0 || str == 0) throw std::invalid_argument("Pooling window/stride must be > 0");
        }
        size_t window() const { return win; }
        size_t stride() const { return str; }
        Mode mode() const { return mode_; }
        Tensor forward(const Tensor& in) const {
            size_t N = in.shape[0], C = in.shape[1], H = in.shape[2], W = in.shape[3];
            if (H < win || W < win) throw std::invalid_argument("Pooling window larger than input");
            size_t oh = (H - win) / str + 1, ow = (W - win) / str + 1;
            Tensor out(N, C, oh, ow);
            for (size_t n = 0; n < N; ++n) for (size_t c = 0; c < C; ++c)
              for (size_t h = 0; h < oh; ++h) for (size_t w = 0; w < ow; ++w) {
                double best = -std::numeric_limits<double>::infinity(), sum = 0.0;
                for (size_t r = 0; r < win; ++r) for (size_t s = 0; s < win; ++s) {
                    double v = in(n, c, h * str + r, w * str + s);
                    best = std::max(best, v); sum += v;
                }
                out(n, c, h, w) = (mode_ == Mode::Max) ? best : sum / double(win * win);
            }
            return out;
        }
    };
}
#endif

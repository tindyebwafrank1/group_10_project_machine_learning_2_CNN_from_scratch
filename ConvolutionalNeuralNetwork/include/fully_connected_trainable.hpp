
#ifndef CNN_FULLY_CONNECTED_TRAINABLE_HPP
#define CNN_FULLY_CONNECTED_TRAINABLE_HPP

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "fully_connected.hpp"

namespace cnn {

// Scoped inside cnn on because it does NOT leak into the global namespace.
using namespace std;

// Helper namespace to handle validation logic
namespace fc_detail {
    inline void fail(const std::string& context, const std::string& msg) {
        throw std::invalid_argument("FullyConnectedTrainable::" + context + ": " + msg);
    }

    inline void requireFinite(const std::vector<double>& data, const std::string& context, const std::string& name) {
        for (double v : data) {
            if (!std::isfinite(v)) {
                throw std::invalid_argument("FullyConnectedTrainable::" + context + ": " + name + " contains NaN or Inf");
            }
        }
    }

    inline void requireValid(const Tensor& t, const std::string& context) {
        if (t.shape.size() != 4) {
            fail(context, "Tensor must be 4-D {N, C, H, W}");
        }
        if (t.data.size() != t.shape[0] * t.shape[1] * t.shape[2] * t.shape[3]) {
            fail(context, "Tensor data size does not match shape");
        }
    }
}

class FullyConnectedTrainable : public FullyConnected {
public:
    /// Same arguments and errors as FullyConnected. Gradients start at zero.
    FullyConnectedTrainable(size_t in_features, size_t out_features)
        : FullyConnected(in_features, out_features),
          grad_weights_(in_features * out_features, 0.0),
          grad_bias_(out_features, 0.0) {}

    Tensor backward(const Tensor& grad_output) {
        if (!hasCachedInput()) {
            throw logic_error("FullyConnectedTrainable::backward: forward() has not been called");
        }
        const Tensor& x = cachedInput();
        fc_detail::requireValid(grad_output, "backward");
        const size_t in = inFeatures();
        const size_t out = outFeatures();
        const size_t batch = x.shape[0];
        if (grad_output.shape[0] != batch) {
            fc_detail::fail("backward", "grad_output has batch size " + to_string(grad_output.shape[0]) +
                                            " but the last forward() input had " + to_string(batch));
        }
        const size_t per_sample = grad_output.data.size() / batch;
        if (per_sample != out) {
            fc_detail::fail("backward", "expected " + to_string(out) + " gradient values per sample but got " +
                                            to_string(per_sample));
        }
        fc_detail::requireFinite(grad_output.data, "backward", "grad_output");

        // Everything is computed into temporaries first, so a failure leaves the layer unchanged.
        const vector<double>& W = weights();
        vector<double> dw(W.size(), 0.0);
        vector<double> db(out, 0.0);
        Tensor dx(x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        dx.data.assign(x.data.size(), 0.0);

        for (size_t n = 0; n < batch; ++n) {
            const double* xin = x.data.data() + n * in;
            const double* delta = grad_output.data.data() + n * out;
            double* dxn = dx.data.data() + n * in;
            for (size_t o = 0; o < out; ++o) {
                const double g = delta[o];
                const double* w = W.data() + o * in;
                double* dwo = dw.data() + o * in;
                db[o] += g;
                for (size_t i = 0; i < in; ++i) {
                    dwo[i] += g * xin[i];
                    dxn[i] += w[i] * g;
                }
            }
        }

        for (double v : dw) requireResult(v);
        for (double v : db) requireResult(v);
        for (double v : dx.data) requireResult(v);

        grad_weights_ = move(dw);
        grad_bias_ = move(db);
        return dx;
    }

    /// dL/dW from the last backward() call, row-major {out_features, in_features}.
    const vector<double>& gradWeights() const noexcept { return grad_weights_; }
    /// dL/db from the last backward() call.
    const vector<double>& gradBias() const noexcept { return grad_bias_; }

    /// Sets both stored gradients to zero.
    void zeroGrad() {
        grad_weights_.assign(grad_weights_.size(), 0.0);
        grad_bias_.assign(grad_bias_.size(), 0.0);
    }

    /// Adds delta_w to the weights and delta_b to the biases. An optimizer such as SGD passes
    /// -learning_rate * gradient. Sizes must match, every value must be finite and the new
    /// parameters must stay finite. On any error nothing changes.
    void applyUpdate(const vector<double>& delta_w, const vector<double>& delta_b) {
        if (delta_w.size() != weights().size()) {
            fc_detail::fail("applyUpdate", "expected " + to_string(weights().size()) +
                                               " weight deltas but got " + to_string(delta_w.size()));
        }
        if (delta_b.size() != bias().size()) {
            fc_detail::fail("applyUpdate", "expected " + to_string(bias().size()) + " bias deltas but got " +
                                               to_string(delta_b.size()));
        }
        fc_detail::requireFinite(delta_w, "applyUpdate", "delta_w");
        fc_detail::requireFinite(delta_b, "applyUpdate", "delta_b");
        vector<double> w = weights();
        vector<double> b = bias();
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] += delta_w[i];
            requireResult(w[i]);
        }
        for (size_t i = 0; i < b.size(); ++i) {
            b[i] += delta_b[i];
            requireResult(b[i]);
        }
        setWeights(w);  // both calls re-check size and finiteness; they cannot fail after the checks above
        setBias(b);
    }

private:
    /// Throws std::runtime_error if a computed value is infinite or NaN.
    static void requireResult(double v) {
        if (!std::isfinite(v)) {
            throw runtime_error("FullyConnectedTrainable: a computed value overflowed to infinity or NaN");
        }
    }

    vector<double> grad_weights_;  // dL/dW from the last backward(), same layout as weights()
    vector<double> grad_bias_;     // dL/db from the last backward()
};

}  // namespace cnn

#endif 

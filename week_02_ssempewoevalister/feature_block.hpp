#ifndef FEATURE_BLOCK_HPP
#define FEATURE_BLOCK_HPP

// Week 2 - Evalister Ssempewo: CNN feature-extraction block builder.
//   FeatureBlock     : Conv2D -> Activation -> Pooling
//   FeatureExtractor : chains several FeatureBlocks (the "builder")
//
// Depends on teammates' modules (see docs/feature_block.md, "Assumed interfaces"):
//   tensor.hpp (Agatha), layers.hpp Conv2D (Joan/Evalister + padding M1 + stride M2),
//   activations.hpp (Jeremy/Agatha), pooling.hpp (Mugarura/Joan).

#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "cnn_library/tensor.hpp"
#include "cnn_library/layers.hpp"
#include "cnn_library/activations.hpp"
#include "cnn_library/pooling.hpp"

namespace CNN {

    enum class Activation { ReLU, Sigmoid, Tanh };

    class FeatureBlock {
    public:
        Conv2D conv;   // public to match the Week 2 interface standard
        Pooling pool;  // default: 2x2 window, stride 2, max. Replace via `block.pool = Pooling(...)`

    private:
        size_t in_c, out_c, k, stride_, pad_;
        Activation act;
        Tensor kernel;

        // Output size of one spatial dimension after the conv step.
        size_t conv_dim(size_t in) const {
            if (in + 2 * pad_ < k)
                throw std::invalid_argument("FeatureBlock: kernel (" + std::to_string(k) +
                                            ") larger than padded input (" + std::to_string(in + 2 * pad_) + ")");
            return (in + 2 * pad_ - k) / stride_ + 1;
        }
        size_t pool_dim(size_t in) const {
            if (in < pool.window())
                throw std::invalid_argument("FeatureBlock: pooling window larger than feature map");
            return (in - pool.window()) / pool.stride() + 1;
        }

        Tensor apply_activation(const Tensor& x) const {
            switch (act) {
                case Activation::ReLU:    return ReLU().forward(x);
                case Activation::Sigmoid: return Sigmoid().forward(x);
                case Activation::Tanh:    return Tanh().forward(x);
            }
            throw std::logic_error("FeatureBlock: unknown activation");
        }

        // He-style normal init: std = sqrt(2 / fan_in). Deterministic for a given seed.
        void init_kernel(unsigned seed) {
            std::mt19937 gen(seed);
            double stddev = std::sqrt(2.0 / double(in_c * k * k));
            std::normal_distribution<double> dist(0.0, stddev);
            for (double& v : kernel.data) v = dist(gen);
        }

    public:
        FeatureBlock(size_t in_channels, size_t out_channels, size_t k_size,
                     size_t stride, size_t pad,
                     Activation activation = Activation::ReLU, unsigned seed = 42)
            : conv(stride, pad), pool(),
              in_c(in_channels), out_c(out_channels), k(k_size), stride_(stride), pad_(pad),
              act(activation), kernel(out_channels, in_channels, k_size, k_size) {
            if (in_c == 0 || out_c == 0 || k == 0)
                throw std::invalid_argument("FeatureBlock: channels and kernel size must be > 0");
            if (stride_ == 0)
                throw std::invalid_argument("FeatureBlock: stride must be > 0");
            init_kernel(seed);
        }

        size_t in_channels() const { return in_c; }
        size_t out_channels() const { return out_c; }
        const Tensor& kernel_weights() const { return kernel; }

        // Replace the filters (e.g. with Mugarura's kernel initializer, or fixed values in tests).
        void set_kernel(const Tensor& new_kernel) {
            if (new_kernel.shape.size() != 4 || new_kernel.shape[0] != out_c || new_kernel.shape[1] != in_c ||
                new_kernel.shape[2] != k || new_kernel.shape[3] != k)
                throw std::invalid_argument("FeatureBlock: kernel shape must be {out_c, in_c, k, k}");
            kernel = new_kernel;
        }

        // Output {C, H, W} for a given input H x W. Throws if the input is too small.
        std::array<size_t, 3> output_shape(size_t H, size_t W) const {
            return {out_c, pool_dim(conv_dim(H)), pool_dim(conv_dim(W))};
        }

        Tensor forward(const Tensor& input) const {
            if (input.shape.size() != 4)
                throw std::invalid_argument("FeatureBlock: input must be a 4D tensor (N, C, H, W)");
            if (input.shape[1] != in_c)
                throw std::invalid_argument("FeatureBlock: expected " + std::to_string(in_c) +
                                            " input channels, got " + std::to_string(input.shape[1]));
            output_shape(input.shape[2], input.shape[3]); // validates sizes with a clear message
            Tensor z = conv.forward(input, kernel);
            Tensor a = apply_activation(z);
            return pool.forward(a);
        }
    };

    // Builder: stacks blocks, inferring each block's input channels from the previous one.
    class FeatureExtractor {
    private:
        std::vector<FeatureBlock> blocks;
        size_t channels;
        unsigned base_seed;

    public:
        explicit FeatureExtractor(size_t input_channels, unsigned seed = 42)
            : channels(input_channels), base_seed(seed) {
            if (input_channels == 0)
                throw std::invalid_argument("FeatureExtractor: input channels must be > 0");
        }

        FeatureExtractor& add_block(size_t out_channels, size_t k_size, size_t stride, size_t pad,
                                    Activation act = Activation::ReLU) {
            blocks.emplace_back(channels, out_channels, k_size, stride, pad, act,
                                base_seed + static_cast<unsigned>(blocks.size()));
            channels = out_channels;
            return *this; // allows chaining
        }

        size_t num_blocks() const { return blocks.size(); }
        FeatureBlock& block(size_t i) { return blocks.at(i); }
        const FeatureBlock& block(size_t i) const { return blocks.at(i); }

        std::array<size_t, 3> output_shape(size_t H, size_t W) const {
            if (blocks.empty()) throw std::logic_error("FeatureExtractor: no blocks added");
            std::array<size_t, 3> s{channels, H, W};
            for (const FeatureBlock& b : blocks) s = b.output_shape(s[1], s[2]);
            return s;
        }

        Tensor forward(const Tensor& input) const {
            if (blocks.empty()) throw std::logic_error("FeatureExtractor: no blocks added");
            Tensor x = blocks[0].forward(input);
            for (size_t i = 1; i < blocks.size(); ++i) x = blocks[i].forward(x);
            return x;
        }
    };
}

#endif // FEATURE_BLOCK_HPP

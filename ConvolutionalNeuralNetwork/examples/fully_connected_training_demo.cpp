#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <algorithm>

#include "../include/fully_connected_backward.hpp"
#include "../include/tensor.hpp"

using namespace std;
using namespace cnn;

static vector<double> softmaxRow(const double* z, size_t k) {
    double mx = z[0];
    for (size_t i = 1; i < k; ++i) mx = max(mx, z[i]);
    vector<double> p(k);
    double sum = 0.0;
    for (size_t i = 0; i < k; ++i) {
        p[i] = exp(z[i] - mx);
        sum += p[i];
    }
    for (double& v : p) v /= sum;
    return p;
}

int main() {
    try {
        // Eight points: class 0 near (-1, -1), class 1 near (1, 1).
        const vector<double> points = {-1.0, -0.8, -0.9, -1.2, -1.1, -0.9, -0.8, -1.0,
                                       1.0,  0.9,  0.8,  1.1,  1.2,  0.8,  0.9,  1.0};
        const vector<size_t> labels = {0, 0, 0, 0, 1, 1, 1, 1};
        const size_t N = labels.size();
        const size_t classes = 2;
        const size_t features = 2;

        // Since fully_connected_backward.hpp provides a functional API, 
        // we manage weights and biases natively using vectors.
        vector<vector<double>> W = {{0.01, -0.02}, 
                                    {0.03,  0.01}}; // small start
        vector<double> b(classes, 0.0);

        // Pre-allocate gradient accumulators for the batch (to avoid re-allocation per epoch)
        vector<vector<double>> accum_dw(classes, vector<double>(features, 0.0));
        vector<double> accum_db(classes, 0.0);

        const double learning_rate = 0.5;
        cout << fixed << setprecision(4);
        cout << "epoch   loss     accuracy\n";

        for (int epoch = 0; epoch <= 30; ++epoch) {
            double total_loss = 0.0;
            size_t correct = 0;

            // 1. Zero out the accumulated gradients at the start of the epoch
            for (size_t c = 0; c < classes; ++c) {
                accum_db[c] = 0.0;
                for (size_t f = 0; f < features; ++f) {
                    accum_dw[c][f] = 0.0;
                }
            }

            // 2. Loop through the batch
            for (size_t n = 0; n < N; ++n) {
                // Extract current point (using pointers avoids allocation)
                const double* x_n = &points[n * features]; 

                // Forward pass (z = W*x + b)
                vector<double> z(classes, 0.0);
                for (size_t c = 0; c < classes; ++c) {
                    z[c] = W[c][0] * x_n[0] + W[c][1] * x_n[1] + b[c];
                }

                // Calculate loss and accuracy
                const vector<double> p = softmaxRow(z.data(), classes);
                total_loss -= log(p[labels[n]]);
                
                size_t best = 0;
                for (size_t c = 1; c < classes; ++c) {
                    if (p[c] > p[best]) best = c;
                }
                if (best == labels[n]) ++correct;

                // Upstream gradient (dL/dz)
                vector<double> delta(classes, 0.0);
                for (size_t c = 0; c < classes; ++c) {
                    delta[c] = (p[c] - (c == labels[n] ? 1.0 : 0.0)) / static_cast<double>(N);
                }

                // Backward pass using the provided functional API
                FullyConnectedGradients grads = fullyConnectedBackward(W, {x_n[0], x_n[1]}, delta);

                // Accumulate gradients
                for (size_t c = 0; c < classes; ++c) {
                    accum_db[c] += grads.d_biases[c];
                    for (size_t f = 0; f < features; ++f) {
                        accum_dw[c][f] += grads.d_weights[c][f];
                    }
                }
            }

            // 3. Update weights after processing the whole batch
            for (size_t c = 0; c < classes; ++c) {
                b[c] -= learning_rate * accum_db[c];
                for (size_t f = 0; f < features; ++f) {
                    W[c][f] -= learning_rate * accum_dw[c][f];
                }
            }

            // Logging
            total_loss /= static_cast<double>(N);
            if (epoch % 5 == 0) {
                cout << setw(5) << epoch << "   " << total_loss << "   " << static_cast<double>(correct) / N << "\n";
            }
        }
    } catch (const exception& e) {
        cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
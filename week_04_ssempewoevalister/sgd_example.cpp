// Mini-batch SGD fitting y = 2x + 1 with a one-weight, one-bias model.
// Shows the pattern the training loop Edrine uses: forward, gradients, optimizer step, repeat.
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include "cnn_library/optimizer.hpp"

int main() {
    const size_t n = 64, batch = 8, epochs = 60;
    std::vector<double> xs(n), ys(n);
    for (size_t i = 0; i < n; ++i) { xs[i] = -1.0 + 2.0 * double(i) / double(n - 1); ys[i] = 2.0 * xs[i] + 1.0; }

    std::vector<double> w = {0.0}, b = {0.0};   // parameters
    std::vector<double> dw(1), db(1);            // gradients
    CNN::SGD opt(0.1, 0.9);

    std::vector<size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 gen(123);

    for (size_t epoch = 1; epoch <= epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), gen);
        double epoch_loss = 0.0;
        for (size_t start = 0; start < n; start += batch) {
            CNN::zero_gradients(dw); CNN::zero_gradients(db);
            for (size_t j = start; j < start + batch; ++j) {
                double x = xs[order[j]], y = ys[order[j]];
                double err = (w[0] * x + b[0]) - y;      // prediction error
                epoch_loss += err * err;
                dw[0] += 2.0 * err * x;                  // dL/dw summed over the batch
                db[0] += 2.0 * err;
            }
            opt.step(w, dw, 1.0 / double(batch));        // average over the batch
            opt.step(b, db, 1.0 / double(batch));
        }
        if (epoch % 10 == 0)
            std::cout << "epoch " << epoch << "  mse " << epoch_loss / double(n)
                      << "  w=" << w[0] << "  b=" << b[0] << "\n";
    }
    std::cout << "Target: w=2, b=1\n";
    return 0;
}

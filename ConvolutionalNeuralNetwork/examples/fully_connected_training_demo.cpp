#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "../include/fully_connected_backward.hpp"

#include "../include/tensor.hpp"

using namespace std;
using namespace cnn;

static vector<double> softmaxRow(const double* z, size_t k) {
    double mx = z[0];
    for (size_t i = 1; i < k; ++i) mx = max(mx, z[i]);  // subtract the maximum for stability
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
        const size_t N = labels.size(), classes = 2;

        Tensor x(N, 2, 1, 1);
        x.data = points;

        FullyConnectedTrainable layer(2, classes);
        layer.setWeights({0.01, -0.02, 0.03, 0.01});  // small start (the real code uses Xavier/He values)

        const double learning_rate = 0.5;
        cout << fixed << setprecision(4);
        cout << "epoch   loss     accuracy\n";

        for (int epoch = 0; epoch <= 30; ++epoch) {
            const Tensor z = layer.forward(x);  // scores, input saved for backward()

            double loss = 0.0;
            size_t correct = 0;
            Tensor delta(N, classes, 1, 1);  // dL/dz
            for (size_t n = 0; n < N; ++n) {
                const vector<double> p = softmaxRow(&z.data[n * classes], classes);
                loss -= log(p[labels[n]]);
                size_t best = 0;
                for (size_t c = 1; c < classes; ++c)
                    if (p[c] > p[best]) best = c;
                if (best == labels[n]) ++correct;
                for (size_t c = 0; c < classes; ++c)
                    delta.data[n * classes + c] = (p[c] - (c == labels[n] ? 1.0 : 0.0)) / static_cast<double>(N);
            }
            loss /= static_cast<double>(N);
            if (epoch % 5 == 0) {
                cout << setw(5) << epoch << "   " << loss << "   " << static_cast<double>(correct) / N << "\n";
            }

            (void)layer.backward(delta);  // stores dL/dW and dL/db, returns dL/dx (not needed here)

            vector<double> dw = layer.gradWeights();  // plain SGD: W <- W - lr * dL/dW
            vector<double> db = layer.gradBias();
            for (double& v : dw) v *= -learning_rate;
            for (double& v : db) v *= -learning_rate;
            layer.applyUpdate(dw, db);
        }
    } catch (const exception& e) {
        cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
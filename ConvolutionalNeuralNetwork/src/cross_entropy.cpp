#include "cross_entropy.hpp"
#include <cmath>
#include <stdexcept>

using namespace std;

double CrossEntropyLoss::forward(const Tensor& predictions, const Tensor& targets) const {
   
    if (predictions.shape != targets.shape) {
        throw invalid_argument("CrossEntropyLoss forward: Shape mismatch between predictions and targets.");
    }


    size_t batch_size = predictions.shape[0];
    size_t num_classes = predictions.shape[1];
    size_t height = predictions.shape[2];
    size_t width = predictions.shape[3];

    double total_loss = 0.0;

    //  Compute loss across the flat contiguous memory arrays
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < num_classes; ++c) {
            for (size_t h = 0; h < height; ++h) {
                for (size_t w = 0; w < width; ++w) {
                    
                    
                    double pred_val = predictions(n, c, h, w);
                    double true_val = targets(n, c, h, w);

                    
                    if (pred_val < epsilon) {
                        pred_val = epsilon;
                    } else if (pred_val > 1.0 - epsilon) {
                        pred_val = 1.0 - epsilon;
                    }

                    total_loss += true_val * log(pred_val);
                }
            }
        }
    }

    // Compute the final average scalar loss across the entire batch size N
    return -total_loss / static_cast<double>(batch_size);
}

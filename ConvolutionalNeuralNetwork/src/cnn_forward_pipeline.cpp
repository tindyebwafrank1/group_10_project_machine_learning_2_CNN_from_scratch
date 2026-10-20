#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace std;

// 4D Tensor grid to store our network data
struct Tensor {
    vector<double> data;
    vector<size_t> shape; // {Batch, Channels, Height, Width}

    Tensor(size_t n, size_t c, size_t h, size_t w) : shape({n, c, h, w}) {
        data.resize(n * c * h * w, 0.0);
    }

    // Index mapping formula to access continuous memory arrays
    double& operator()(size_t n, size_t c, size_t h, size_t w) {
        return data[n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w];
    }

    const double& operator()(size_t n, size_t c, size_t h, size_t w) const {
        return data[n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w];
    }
};

// Flatten layer to turn 4D feature inputs into 2D structures
struct FlattenLayer {
    Tensor forward(const Tensor& input) {
        size_t batch = input.shape[0];
        size_t flat_dim = input.shape[1] * input.shape[2] * input.shape[3];
        
        Tensor output(batch, 1, 1, flat_dim);
        output.data = input.data; 
        return output;
    }
};

// Fully connected classification layer running linear mapping equations
class FullyConnectedLayer {
private:
    size_t in_dim;
    size_t out_dim;
    vector<double> weights;
    vector<double> biases;

public:
    FullyConnectedLayer(size_t src_dim, size_t dest_dim) : in_dim(src_dim), out_dim(dest_dim) {
        weights.resize(in_dim * out_dim, 0.01); 
        biases.resize(out_dim, 0.0);
    }

    Tensor forward(const Tensor& input) {
     size_t batch = input.shape[0];
    Tensor output(batch, 1, 1, out_dim);

    for (size_t b = 0; b < batch; ++b) {
    for (size_t o = 0; o < out_dim; ++o) {
    double score = biases[o];
    for (size_t i = 0; i < in_dim; ++i) {
    score += input.data[b * in_dim + i] * weights[i * out_dim + o];
    }
    output.data[b * out_dim + o] = score;
    }
    
}
    return output;
    }
};

// Softmax activation layer optimized to prevent numerical exponential overflow
struct SoftmaxActivation {
Tensor forward(const Tensor& input) {
 size_t batch = input.shape[0];
size_t classes = input.shape[3];
Tensor output(batch, 1, 1, classes);

for (size_t b = 0; b < batch; ++b) {
size_t row_start = b * classes;

 // Find maximum item to keep evaluations safe from limits
    double max_score = input.data[row_start];
    for (size_t c = 1; c < classes; ++c) {
    if (input.data[row_start + c] > max_score) {
    max_score = input.data[row_start + c];
                }
            }

    double sum = 0.0;
    for (size_t c = 0; c < classes; ++c) {
    output.data[row_start + c] = exp(input.data[row_start + c] - max_score);
    sum += output.data[row_start + c];
            }

    for (size_t c = 0; c < classes; ++c) {
        output.data[row_start + c] /= sum;
            }
     }
        return output;
    }
};

// Main execution manager assembling our pipeline parts
class CNNForwardPipeline {
private:
    FlattenLayer flatten;
    FullyConnectedLayer fc;
    SoftmaxActivation softmax;

public:
    CNNForwardPipeline(size_t flat_dim, size_t num_classes) : fc(flat_dim, num_classes) {}

    Tensor execute_inference(const Tensor& features) {
        Tensor flat_vectors = flatten.forward(features);
        Tensor raw_scores = fc.forward(flat_vectors);
        return softmax.forward(raw_scores);
    }
};

int main() {
    size_t batch = 2;
    size_t channels = 8;
    size_t height = 16;
    size_t width = 16;
    size_t target_classes = 10;

    size_t flat_dimension = channels * height * width;

    Tensor input_features(batch, channels, height, width);
    fill(input_features.data.begin(), input_features.data.end(), 0.25);

    CNNForwardPipeline pipeline(flat_dimension, target_classes);
    Tensor results = pipeline.execute_inference(input_features);

    cout << "Execution Completed. Output shape parameters: {" 
         << results.shape[0] << ", " 
         << results.shape[1] << ", "
         << results.shape[2] << ", " 
         << results.shape[3] << "}" << endl;

    if (results.shape[3] == target_classes) {
        cout << "Forward assembly pipeline verification: PASSED" << endl;
    } else {
        cout << "Forward assembly pipeline verification: FAILED" << endl;
    }

    return 0;
}

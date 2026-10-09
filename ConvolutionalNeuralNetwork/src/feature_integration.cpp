#include <iostream>
#include <vector>
#include <cstddef>

// Allows the code to use standard objects like vector and cout without the std:: prefix
using namespace std;

// This class represents a 4D image data block holding values in a 1D vector array
class Tensor {
public:
    vector<double> data;
    vector<size_t> shape; // Stores the configuration: {Batch, Channels, Height, Width}

    // Creates the tensor layout and sizes up the data memory storage block
    Tensor(size_t n, size_t c, size_t h, size_t w) : shape({n, c, h, w}) {
        data.resize(n * c * h * w, 0.0);
    }

    // Mathematical formula to convert 4D positions into a single 1D index
    double& operator()(size_t n, size_t c, size_t h, size_t w) {
        return data[n * (shape[1] * shape[2] * shape[3]) + c * (shape[2] * shape[3]) + h * shape[3] + w];
    }
};

// Simulated convolution layer class used to resize spatial dimensions
class Conv2D {
private:
    size_t in_c, out_c, k_size, stride, padding;
public:
    Conv2D() : in_c(0), out_c(0), k_size(0), stride(0), padding(0) {}
    Conv2D(size_t ic, size_t oc, size_t k, size_t s, size_t p) 
        : in_c(ic), out_c(oc), k_size(k), stride(s), padding(p) {}

    // Calculates the new image dimensions and outputs a transformed data tensor
    Tensor forward(const Tensor& input) {
        size_t h_out = ((input.shape[2] - k_size + 2 * padding) / stride) + 1;
        size_t w_out = ((input.shape[3] - k_size + 2 * padding) / stride) + 1;
        
        Tensor output(input.shape[0], out_c, h_out, w_out);
        for (size_t i = 0; i < output.data.size(); ++i) {
            output.data[i] = input.data[i % input.data.size()] * 0.5;
        }
        return output;
    }
};

// Simulated pooling layer class used to downsample data fields
class Pooling {
public:
    // Shrinks the width and height dimensions by half to condense the features
    Tensor forward(const Tensor& input) {
        size_t h_out = input.shape[2] / 2;
        size_t w_out = input.shape[3] / 2;
        Tensor output(input.shape[0], input.shape[1], h_out ? h_out : 1, w_out ? w_out : 1);
        for (size_t i = 0; i < output.data.size(); ++i) {
            output.data[i] = input.data[i % input.data.size()];
        }
        return output;
    }
};

// Main pipelines manager class that links layers together
class FeatureBlock {
public:
    Conv2D conv;
    Pooling pool;

    // Direct constructor that sets up the internal convolution and pooling layers
    FeatureBlock(size_t in_c, size_t out_c, size_t k_size, size_t stride, size_t pad)
        : conv(in_c, out_c, k_size, stride, pad), pool() {}

    // Coordinates data routing by passing it through Conv, then ReLU, and finally Pooling
    Tensor forward(const Tensor& input) {
        Tensor conv_out = conv.forward(input);

        // Simple ReLU loop that turns any negative pixel activation value into zero
        for (size_t i = 0; i < conv_out.data.size(); ++i) {
            if (conv_out.data[i] < 0.0) conv_out.data[i] = 0.0;
        }

        return pool.forward(conv_out);
    }
};

// Program entry point to test our connected framework logic
int main() {
    cout << "=== Pipeline Integration Execution ===" << endl;

    // Creates a mock input sample and passes it through our pipeline blocks
    Tensor img_input(1, 3, 14, 14);
    FeatureBlock blocks(3, 6, 3, 1, 1);
    Tensor result = blocks.forward(img_input);

    // Prints the final height and width of the processed data to verify correctness
    cout << "Processed Output Spatial Block: " << result.shape[2] << "x" << result.shape[3] << endl;
    return 0;
}

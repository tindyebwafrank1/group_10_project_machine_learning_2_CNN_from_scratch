                FINAL PROJECT REPORT

                    GROUP 10

         CONVOLUTIONAL NEURAL NETWORK  (CNN) FROM SCRATCH

## PROBLEM ADDRESSED:
 Implementing a Convolutional Neural Network (CNN) entirely from scratch in pure C++ without relying on external deep learning frameworks (like TensorFlow or PyTorch).

## PROJECT OBJECTIVES
  * Design a modular, reusable C++ library (cnn_lib) with a clean public API (include/).
  * Apply Object-Oriented Programming (OOP) principles, raw memory management, and first-principles mathematical formulas.
  * Rigorously test all components using 33 dedicated unit tests via CMake and CTest.



## LIBRARY ARCHITECTURE AND DIRECTORY STRUCTURE
Our project follows a professional directory layout, separating public interfaces from internal implementations:

ConvolutionalNeuralNetwork/
├── README.md
├── CMakeLists.txt
├── include/
│       └──tensor.hpp
│       ├── conv2d.hpp
│       ├── pooling.hpp
│       ├── activation.hpp
│       ├── fully_connected.hpp
│       ├── softmax.hpp
│       └── loss.hpp
├── src/
│   └── ... (.cpp implementations)
├── tests/
│   └── ... (33 unit test suites)
├── examples/
│   └── cnn_final_app.cpp
└── reports/
    ├── week-01.md to week-04.md
    └── final-report.md


## Class Deconstruction & Implementation Details


## Module A: 
The Core Tensor Class (Tensor)
1. **What class did you create?** `Tensor` (representing 4D data blocks: $N \times C \times H \times W$).
2. **Why?** Raw images and feature maps must be stored in contiguous memory blocks with efficient multidimensional indexing to handle batch size, channels, height, and width.
3. **What data does it hold?**
   * `std::vector<double> data`: Contiguous 1D vector storing all floating-point values.
   * `std::vector<size_t> shape`: Dimensions vector `{N, C, H, W}`.
4. **What functions operate on that data?**
   * Constructor `Tensor(size_t n, size_t c, size_t h, size_t w)`: Allocates memory based on shape.
   * `double& operator()(size_t n, size_t c, size_t h, size_t w)`: Provides multi-index element access.
   * Flattening and gradient allocation helpers.
5. **Why are some members private and others public?** 
   * *Public:* Constructor and indexing operators (`operator()`) are public to allow seamless interaction from convolutional and pooling layers.
   * *Private/Protected:* The underlying raw `data` vector is encapsulated to maintain safe data structures while offering inline performance accessors.

### Module B: Two-Dimensional Convolution (`Conv2D`)
1. **What class did you create?** `Conv2D` (handles multi-channel convolution, kernels, strides, and zero-padding).
2. **Why?** To extract spatial features from input tensors using learnable filter weights.
3. **What data does it hold?**
   * `Tensor weights`: Filter kernels of shape $(out\_channels, in\_channels, k\_h, k\_w)$.
   * `Tensor biases`: Bias terms for each output channel.
   * `size_t stride`, `size_t padding`: Spatial hyperparameters.
4. **What functions operate on that data?**
   * `Tensor forward(const Tensor& input)`: Computes sliding-window dot products across channels.
   * `void backward(...)`: Computes gradients with respect to weights, biases, and inputs for backpropagation.
5. **Why are some members private and others public?**
   * *Public:* `forward()` and `backward()` form the public API used by the feature extraction pipeline.
   * *Private:* Hyperparameters and intermediate cache states (like padded inputs) are kept private to encapsulate internal math state.

### Module C: Pooling Layers (`Pooling`)
1. **What class did you create?** `Pooling` (supports Max Pooling and Average Pooling).
2. **Why?** To downsample feature maps, reduce spatial dimensions, and provide translational invariance.
3. **What data does it hold?**
   * `size_t pool_size`, `size_t stride`: Downsampling configuration.
   * `std::string mode`: Distinguishes between `"max"` and `"average"`.
4. **What functions operate on that data?**
   * `Tensor forward(const Tensor& input)`: Scans windows and extracts max or average values (while tracking max indices for backprop).
   * `Tensor backward(...)`: Routes gradients back to the exact positions where maximum values originated (for Max Pooling).
5. **Why are some members private and others public?**
   * *Public:* Layer configuration and forward/backward execution methods.
   * *Private:* Index tracking matrices used during the backward pass.

### Module D: Fully Connected Layer (`FullyConnected`)
1. **What class did you create?** `FullyConnected` (Dense layer).
2. **Why?** To map flattened feature representations into final class logits for classification.
3. **What data does it hold?**
   * `Tensor weights`, `Tensor biases`.
   * Cached inputs and outputs for gradient calculation.
4. **What functions operate on that data?**
   * `Tensor forward(const Tensor& input)`: Matrix multiplication ($z = Wx + b$).
   * `Tensor backward(const Tensor& output_grad)`: Computes weight/bias gradients and propagates error backward.
5. **Why are some members private and others public?**
   * *Public:* Matrix-vector interface methods.
   * *Private:* Weight initialization states and intermediate activation caches.

---

## 4. Testing Approach & Results
* **Framework:** Custom C++ unit testing framework compiled via CMake and run using CTest.
* **Test Coverage:** **33 dedicated test files** covering every single module:
  * Tensor allocation and indexing (`test_tensor`)
  * Padding and strides (`test_padding`, `test_strides`)
  * Convolutions and multi-channel pipelines (`test_convolution`, `test_multi_channel_convolution`)
  * Activations and pooling (`test_relu`, `test_max_pooling`, `test_average_pooling`)
  * Dense layers, Softmax, and Cross-Entropy loss (`test_fully_connected`, `test_softmax`, `test_cross_entropy`)
  * Complete backpropagation and SGD optimization (`test_training`, `test_convolution_backward`)
* **Result:** **100% pass rate (33/33 tests passing successfully in 1.73 seconds)**.

---

## 5. Team Contributions & Collaboration
* **Frank (Team Lead):** Dataset parsing, image loading engine, loss gradients & Softmax backprop.
* **Peter (Operations Lead):** Image preprocessing normalization, stride mechanics, Fully Connected forward pass.
* **Jeremy:** Image representation structures, ReLU activation, Xavier/He weight initialization, flattening backward routing.
* **Agatha:** Tensor vector math, Sigmoid/Tanh activations, Softmax numerical stability.
* **Mugarura:** Filter initialization, Max Pooling layer & index tracking, Cross-Entropy loss.
* **Joan:** 2D convolution sliding window math, Average Pooling, Argmax prediction logic, Conv2D backward pass.
* **Evarista:** Multi-channel convolution passes, Feature extraction block builders, Evaluation metrics suite, SGD optimizer.
* **Edrine (Integration Lead):** Convolution pipeline integration, Week 1-2 demos, Mini-batch training loop & final application packaging.

---

## 6. AI Use Declaration
* **Tool:** Advanced AI assistant (Gemini)
* **Purpose:** Architecture brainstorming, debugging memory boundaries in convolution loops, reviewing C++ tensor indexing formulas, and structuring documentation guidelines.
* **Reason:** To accelerate syntax verification and ensure adherence to clean C++ design patterns while maintaining full team comprehension of all code written.
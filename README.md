

# Convolutional Neural Network (CNN) From Scratch in C++
**Group 10 | Machine Learning 2 (ML2)**

## 1. Problem Being Addressed
Modern deep learning frameworks (like TensorFlow and PyTorch) abstract away the underlying mathematics and memory management, acting as black boxes. The challenge of this project was to demystify neural networks by building a Convolutional Neural Network entirely from scratch in pure C++ without relying on any external machine learning libraries. This required implementing everything from raw contiguous memory tensor allocation to the complex calculus of backpropagation and gradient descent.

## 2. Project Objectives
*   **Architectural Integrity:** Design a modular, highly reusable static library (`cnn_lib`) with a strictly separated public API (`include/`) and internal implementations (`src/`).
*   **First-Principles Engineering:** Implement object-oriented principles, contiguous memory allocation, and mathematical formulations without external matrix libraries.
*   **Rigorous Validation:** Verify every component using 33 dedicated unit tests driven by CMake and CTest.

## 3. Library Design
The project adheres to professional C++ directory conventions, ensuring that the core logic is entirely decoupled from the test suites and demonstration programs.

```text
ConvolutionalNeuralNetwork/
├── CMakeLists.txt
├── include/                  # Public API headers
│   ├── tensor.hpp
│   ├── conv2d.hpp
│   ├── pooling.hpp
│   ├── fully_connected.hpp
│   └── ...
├── src/                      # Internal C++ implementations
│   └── ...
├── tests/                    # 33 Unit Test suites
│   └── ...
└── examples/                 # Demonstration applications
    └── ...
```
The library utilizes a **Functional API** for its mathematical operations (e.g., `fullyConnectedBackward` takes weights, inputs, and gradients as arguments and returns the computed gradients). This stateless design ensures the library is thread-safe, predictable, and highly reusable. Complex structures, like the `FullyConnected` layer, utilize strict encapsulation to protect internal weights and dimensions.

## 4. Main Functionality
The library provides reusable C++ components for core CNN operations:
*   **Tensor Operations:** Dynamic N-dimensional representation, contiguous memory allocation, indexing, and numerical safety.
*   **2D Convolution:** Multi-channel convolution, padding, stride handling, and spatial feature extraction.
*   **Pooling:** Max and average pooling to downsample feature maps and provide translational invariance.
*   **Dense Layers:** Flattening feature maps and performing weight-based matrix/vector calculations (\(z = Wx + b\)).
*   **Activations & Loss:** ReLU, Sigmoid, Softmax, and Cross-Entropy loss.
*   **Training:** A fully functional mini-batch training loop utilizing SGD (Stochastic Gradient Descent).

## 5. Important Implementation Decisions
*   **Memory Management:** Tensors are stored as a contiguous 1D `std::vector<double>`, with multi-index access handled via an overloaded `operator()(n, c, h, w)`.
*   **Encapsulation:** In the `FullyConnected` class, weights and biases are strictly private. They can only be mutated via validated setters (`setWeights`), which check for dimension mismatches and NaN/Infinity values before committing changes.
*   **Const-Correctness:** The library distinguishes between `forward()` (used in training, mutates the object to cache input) and `infer()` (used in evaluation, marked `const` to guarantee it cannot alter the model).
*   **Zero-Padding Without Memory Copies:** Convolution and stride logic map padded coordinates back to the original image mathematically, avoiding the overhead of physically allocating a larger padded matrix.

## 6. Demonstration of the Working Library
The library is compiled using CMake and MinGW/GCC. It builds cleanly with strict adherence to C++17 standards (`-Wall -Wextra -Wpedantic`).

### Build Instructions
To compile the library, tests, and examples, run the following commands from the project root:

```powershell
# 1. Configure CMake
cmake -S ConvolutionalNeuralNetwork -B build -G "MinGW Makefiles"

# 2. Build the project
cmake --build build
```

### Run the Demos
Once built, you can run the included demonstrations. The `fully_connected_training_demo` shows the network successfully learning an 8-point dataset over 30 epochs, reducing loss and improving accuracy.

```powershell
.\build\stride_demo.exe
.\build\fully_connected_training_demo.exe
```

## 7. Testing Approach and Results
*   **Framework:** A custom C++ unit testing framework compiled via CMake and executed using CTest.
*   **Coverage:** 33 dedicated test files covering every module (Tensor allocation, convolutions, pooling, backpropagation, and optimizer updates).
*   **Validation:** Tests include hand-calculated reference matrices, boundary indexing checks, and numerical stability guards.
*   **Result:** **100% pass rate.**

To run the tests yourself:
```powershell
ctest --test-dir build
```
*(Output: `100% tests passed, 0 tests failed out of 33`)*

## 8. Challenges Encountered and How They Were Addressed
*   **The "Most Vexing Parse":** During the integration of the training demos, the compiler misinterpreted `FullyConnectedTrainable layer(2, classes);` as a function declaration rather than a variable instantiation. This was resolved by switching to uniform initialization (`layer{2, classes}`).
*   **API Discrepancies:** The demo files originally relied on a stateful Object-Oriented API that didn't match the functional headers. We bridged this gap by writing manual accumulation loops in the demos, ensuring the gradients calculated by the functional backward pass were correctly applied to the weights.
*   **Windows Security Blocks:** CTest initially threw `BAD_COMMAND` errors because Windows Smart App Control blocked the newly compiled `.exe` files. This was resolved by configuring Windows Defender and utilizing CMake generator expressions (`$<TARGET_FILE:...>`) to ensure accurate test execution paths.

## 9. Individual Contributions
*   **Frank (Team Lead):** Dataset parsing, image loading engine, loss gradients, Softmax backprop.
*   **Peter (Operations Lead):** Image preprocessing normalization, stride mechanics, Fully Connected forward pass.
*   **Jeremy:** Image representation structures, ReLU activation, Xavier/He weight initialization, flattening backward routing.
*   **Agatha:** Tensor vector math, Sigmoid/Tanh activations, Softmax numerical stability.
*   **Mugarura:** Filter initialization, Max Pooling layer & index tracking, Cross-Entropy loss.
*   **Joan:** 2D convolution sliding window math, Average Pooling, Argmax prediction logic, Conv2D backward pass.
*   **Evarista:** Multi-channel convolution passes, Feature extraction block builders, Evaluation metrics suite, SGD optimizer.
*   **Edrine (Integration Lead):** Convolution pipeline integration, Week 1-2 demos, Mini-batch training loop & final application packaging.

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for the full license text.

Copyright (c) 2026 Group 10
## Acknowledgements

Developed as a group project for **Machine Learning 2 (ML2)**. The project
repository and documentation reflect the team's own
implementation, testing, and engineering decisions. We would like to acknowledge the collaborative efforts and specific contributions of every team member

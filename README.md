# CNN From Scratch in C++ 

## 1. Project name and description
This project is an optimized Convolutional Neural Network (CNN) framework built entirely from scratch using pure C++. It is designed specifically for academic evaluation and lightweight deep learning engineering applications, allowing developers and students to understand the low-level linear algebra, structural layouts, and numerical mechanics of neural layers without relying on heavy external tensor libraries.

## 2. Project objectives
The primary objective of this library is to deliver a fully operational, mathematically verified forward and backward CNN execution pipeline by November 2. The project serves to practice advanced C++ object-oriented software patterns, explicit memory alignment techniques, pointer-based multidimensional matrix indexing, cache-friendly iteration loops, and reliable unit testing paradigms.

## 3. Main features
Our library features a weekly modular architecture building toward a complete training lifecycle:
- Week 1 (Current): High-dimensional Tensors, matrix loading mechanisms, multi-channel multi-filter convolution layers, and automated verification suites.
- **My Dedicated Implementation Component**: Multi-channel, multi-filter 2D convolution operations processed over complex batches (`cnn::conv2d_multichannel`).
- Upcoming tasks: Max/Average Pooling blocks, Softmax layer distributions, backpropagation gradients, and gradient descent optimization routines.

## 4. Project structure
Below is our current workspace directory distribution layout:
```text
Group-10---Imbattables/
├── cnn/              # Core tracking architecture folder
│   ├── multichannel_conv.hpp # Declaration signature headers for layer logic
│   └── tensor.hpp            # Structural temporary tensor data wrapper
├── reports/          # Weekly individual and group status report sheets
│   └── week-01.md    # Week 1 project logs and contribution items
├── multichannel_conv.cpp      # Core loop arithmetic implementation code
├── test_multichannel_conv.cpp # Automated assertions test verification block
└── README.md         # Framework documentation manual
```

## 5. Requirements and dependencies
- **Compiler**: GCC `g++` supporting standard C++17 configurations or newer.
- **Build Infrastructure**: CMake version 3.14 or higher.
- **Operating Systems**: Verified across Linux environments (Ubuntu LTS) and native cloud developer environments via GitHub Codespaces.

## 6. Build / install instructions
To set up and compile the overarching project cleanly on a fresh workstation environment, navigate to the root directory and execute the primary build framework:
```bash
cmake -S . -B build
cmake --build build
```
For individual component validation or optimization targets, the codebase can be built natively using direct manual compiler strings:
```bash
g++ -O3 -Wall -Wextra -std=c++17 multichannel_conv.cpp test_multichannel_conv.cpp -o cnn_app
```

## 7. How to use the library
To use this component, include the header file, define your target input/filter dimensions, construct the tensors, and pass them to the execution method:

```cpp
#include "cnn/multichannel_conv.hpp"
#include "cnn/tensor.hpp"

int main() {
    // Instantiate an item matching the tracking dimensions
    Tensor input_tensor(1, 3, 32, 32);  // Batch size 1, 3 Channels, 32x32 dimensions
    Tensor filter_tensor(8, 3, 5, 5);  // 8 Filters, 3 Channels, 5x5 dimensions
    
    // Process the multi-channel 2D calculation layers
    Tensor result = cnn::conv2d_multichannel(input_tensor, filter_tensor);
    return 0;
}
```

### Understanding the Tensor NCHW Layout
Tensors inside this framework are packed linearly into continuous 1D storage arrays using the standard **NCHW** layout convention:
- **N**: Batch index selection
- **C**: Channel index assignment
- **H**: Vertical height offset position
- **W**: Horizontal width index tracking

To translate high-dimensional coordinates into a physical continuous memory location pointer, we apply the 1D index formula mapping:
\[\text{Index} = (n \times C \times H \times W) + (c \times H \times W) + (h \times W) + w\]

## 8. Examples
- **`cnn_app` (Direct Manual Build Binary Output)**: Demonstrates mock multi-channel image input tensors interacting with multi-filter weights, evaluating output size and structural matrix scaling limits.

## 9. How to run the tests
To run the specialized validation checks for the multi-channel convolution functionality, run this combination string within your workspace terminal prompt:
```bash
g++ -std=c++17 -I. multichannel_conv.cpp test_multichannel_conv.cpp -o test_runner && ./test_runner
```
A successful validation verification check will run all underlying math assertions and output:
```text
All multichannel conv tests passed.
```

## 10. Limitations
Our Week 1 framework operates under the following design constraints:
- Spatial Reductions: Valid convolution boundaries only (no padding layers or adjustable strides are supported yet; these are slated for Week 2 extensions).
- Compute Target: Executed entirely on single-threaded CPU architectures.
- **Precision Limits**: Leverages standard double precision data representations across internal layers.

## 11. Contributions by group members
- Ssempewo Evalister (the Author): Formulated, coded, and verified the multi-channel, multi-filter 2D convolution algorithm layer running across input batch flows.
- *Note: Group mates  entries to be cross-referenced and appended by team members inline with corresponding module ownership profiles.*

## 12. How we collaborated
We coordinated development goals utilizing standard branch isolations across individual module tasks. Synchronization checks are handled via closed-loop milestone tracking, explicit Peer Review validation requests on GitHub before merging to main, and collaborative group debugging channels to systematically clear terminal layout blocking bugs.

## 13. AI use
- **Tool**: Claude  AI
- **Application Scope**: Assisted during the development phase with compiler syntax error resolution tracking.
-
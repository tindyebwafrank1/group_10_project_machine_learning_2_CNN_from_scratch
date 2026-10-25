Module 8: System Integration & Demo Pipeline Manager

 1. Overview

This C++ module provides the **system integration layer** for a CNN (Convolutional Neural Network) project.

It connects the different CNN components into one execution pipeline:

1. Load image data.
2. Preprocess/normalize the data.
3. Initialize convolution weights.
4. Perform convolution.
5. Display diagnostic information.

The module also contains automated validation tests for tensor boundary checking and convolution arithmetic.

 2. Required Header Files

#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <string>

Purpose of each header

- `<iostream>` — Provides `cout` and `cerr` for console output.
- `<vector>` — Provides the `vector` container used to store tensor data and dimensions.
- `<cassert>` — Provides `assert()` for checking conditions during program execution.
- `<stdexcept>` — Provides exceptions such as `out_of_range`.
- `<string>` — Provides the `string` data type.

using namespace std;

This allows the program to use names such as `vector`, `string`, and `cout` without writing `std::` each time.

3. Tensor Class

class Tensor {
public:
    vector<double> data;
    vector<size_t> shape;

The `Tensor` class represents a **4-dimensional block of numerical data**.

A CNN commonly represents image data using dimensions such as:

{N, C, H, W}

where:

- `N` = number of images/batch size
- `C` = number of channels
- `H` = height
- `W` = width

For example:

{1, 3, 32, 32}

means:
- 1 image
- 3 channels (e.g. RGB)
- 32 pixels high
- 32 pixels wide
}
 4. Tensor Data Storage

vector<double> data;
vector<size_t> shape;

 `data`
Stores all tensor values in a **one-dimensional vector**.

Although the tensor is logically 4D, the values are stored continuously in memory as a 1D array.

 `shape`
Stores the four dimensions of the tensor:

[N, C, H, W]

 5. Tensor Constructor
Tensor(size_t n, size_t c, size_t h, size_t w)
    : shape({n, c, h, w}) {

    data.resize(n * c * h * w, 0.0);
}

The constructor receives the four dimensions and creates the tensor.

For example:

Tensor image(1, 3, 28, 28);

creates a tensor with:

N = 1
C = 3
H = 28
W = 28

The total number of elements is:

1 × 3 × 28 × 28 = 2352

All elements are initially set to:

0.0

 6. Tensor Index Calculation

The tensor uses the following formula to convert a 4D coordinate into a 1D position:

Index(n,c,h,w)
= n(C×H×W)
+ c(H×W)
+ hW
+ w

The corresponding code is:
return n * (shape[1] * shape[2] * shape[3])
     + c * (shape[2] * shape[3])
     + h * shape[3]
     + w;

This is important because `data` is a one-dimensional vector.

 Example

For:

Shape = {1, 1, 3, 3}

the element:
(0,0,1,2)
has index:
0(1×3×3) + 0(3×3) + 1(3) + 2
= 5

Therefore:
data[5]
contains the value at:
(0,0,1,2)

 7. Boundary Checking

Before calculating the index, the program checks whether the requested coordinates are valid:

if (n >= shape[0] ||
    c >= shape[1] ||
    h >= shape[2] ||
    w >= shape[3]) {
    throw out_of_range(
        "Tensor index out of bounds exception."
    );
}

This prevents the program from accessing memory outside the tensor.

For example:
Tensor test(1, 1, 4, 4);
test(1, 0, 0, 0);
is invalid because the only valid `n` value is:
n = 0
The program therefore throws an `out_of_range` exception.

 8. Tensor Element Access
The class overloads `operator()`:
double& operator()(size_t n, size_t c, size_t h, size_t w) {
    return data[get_index(n, c, h, w)];
}
This allows tensor elements to be accessed naturally:

tensor(0, 0, 2, 3)

instead of manually calculating:

tensor.data[tensor.get_index(0, 0, 2, 3)]

Because the function returns a reference (`double&`), values can also be modified:
tensor(0, 0, 0, 0) = 5.0;

 9. Const Tensor Accessor

The class also provides:
const double& operator()(size_t n, size_t c, size_t h, size_t w) const

This version is used when the tensor is `const`.

It allows values to be read without allowing them to be modified.

This is useful in functions that should only read tensor data.

 10. PipelineIntegrationManager

class PipelineIntegrationManager {
public:

This class coordinates the different CNN components.
Its main responsibility is to make the separate modules work together as one pipeline.
}
 11. Function Pointer Types

The manager defines four function types:

typedef Tensor (*ImageLoaderEngine)(const string&);
ImageLoaderEngine

A function that:
- receives a file path
- loads image data
- returns a `Tensor`

typedef void (*DataPreprocessorEngine)(Tensor&);

 DataPreprocessorEngine

A function that receives a tensor by reference and modifies it.

For example, it could normalize pixel values:
0–255  →  0–1
typedef Tensor (*WeightsInitializerEngine)
    (size_t, size_t, size_t, size_t);

typedef Tensor (*ConvolutionMathEngine)
    (const Tensor&, const Tensor&);

ConvolutionMathEngine
Receives:

input tensor
kernel tensor

and returns the resulting feature map.

These function pointer types allow the integration module to work with implementations created by other developers/modules.

 12. System Validation Tests

The function:

static bool run_system_validation_tests(
    ConvolutionMathEngine architecture_conv)

runs automated checks before the main pipeline is executed.
It returns:

true  → tests passed
false → a test failed

 13. Test 1 — Boundary Protection

The test creates:

Tensor boundary_test(1, 1, 4, 4);
The valid indices are:
N: 0
C: 0
H: 0–3
W: 0–3

The test deliberately tries:

boundary_test(1, 0, 0, 0);
Since `n = 1` is outside the valid range, an exception should occur.

The code catches it:
catch (const out_of_range&) {
    cout << " -> Test 1 ... PASSED\n";
}

This confirms that tensor boundary protection works.

 14. Test 2 — Convolution Arithmetic

The test creates:
Tensor mock_in(1, 1, 3, 3);
Tensor mock_k(1, 1, 2, 2);

The input tensor is filled with:
1 1 1
1 1 1
1 1 1

The kernel is filled with:
2 2
2 2

The first convolution position produces:

(1×2) + (1×2)
+ (1×2) + (1×2)

Therefore: 2 + 2 + 2 + 2 = 8

The program checks:
if (abs(mock_out(0, 0, 0, 0) - 8.0) > 1e-9)

If the result is approximately `8.0`, the convolution test passes.

15. Main System Pipeline

The main pipeline is implemented by:

static void execute_system_pipeline(...)

It receives:
target_file_path
load_stage
preprocess_stage
weight_init_stage
conv_processing_stage

These represent the four main processing stages.

 16. Stage 1 — Image Loading

Tensor processing_tensor = load_stage(target_file_path);

The image loader receives the target file path and returns a tensor containing the image data.

Conceptually:

Image file
    ↓
Image Loader
    ↓
Input Tensor

17. Stage 2 — Data Preprocessing

preprocess_stage(processing_tensor);

The preprocessing function modifies the input tensor.

Typical preprocessing operations may include:

- normalization
- scaling
- reshaping
- channel conversion

Conceptually:


Input Tensor
     ↓
Preprocessing
     ↓
Normalized Tensor
The program then checks that memory was allocated:

assert(processing_tensor.data.size() > 0);

 18. Stage 3 — Weight Initialization

Tensor kernel_filters =
    weight_init_stage(4, 3, 3, 3);

This creates convolution filters with the requested dimensions.
The dimensions are:

4 filters
3 input channels
3 × 3 kernel


So the kernel shape is:

{4, 3, 3, 3}

 19. Stage 4 — Convolution

Tensor feature_maps =
    conv_processing_stage(
        processing_tensor,
        kernel_filters
    );

The convolution engine receives:

Input Tensor
+
Kernel Filters
and produces:

Feature Maps

Conceptually:

Image
  ↓
Preprocessing
  ↓
Convolution
  ↓
Feature Maps

 20. Pipeline Metrics

After processing, the program prints the tensor layouts:


Input Tensor Layout
Kernel Tensor Layout
Output Feature Layout

For example:
Input Tensor Layout   : {1, 3, 32, 32}
Kernel Tensor Layout  : {4, 3, 3, 3}
Output Feature Layout : {1, 4, 30, 30}

The program also prints the first feature-map value:

feature_maps(0, 0, 0, 0)

This provides a simple diagnostic value for checking that the convolution stage produced data.

 21. Exception Handling

The pipeline is protected by:
try {
    ...
}
catch (const exception& e) {
    cerr << "..." << e.what();
}

If an error occurs, the exception is caught and an error message is displayed instead of allowing the program to terminate unexpectedly.

 22. Overall Architecture

The complete system can be viewed as:

CNN SYSTEM INTEGRATION
                     │
                     ▼
              ┌──────────────┐
            │ Image Loader │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ Preprocessor │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ Weight Init. │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ Convolution  │
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │ Feature Maps │
              └──────────────┘
 23. Key Concepts Demonstrated

This module demonstrates several important C++ and CNN concepts:

| Concept | Purpose |

| `vector` | Stores tensor data |
| 4D tensor | Represents CNN data |
| Index mapping | Converts 4D coordinates to 1D memory |
| Operator overloading | Provides `tensor(n,c,h,w)` access |
| Exceptions | Handles invalid tensor access |
| Function pointers | Connects independent modules |
| `assert()` | Performs internal validation |
| `try/catch` | Handles runtime errors |
| Convolution | Produces CNN feature maps |
| Pipeline integration | Connects all processing stages |

The **Module 8 Integration Manager** acts as the central coordinator of the CNN system.

The `Tensor` class provides safe 4D data storage and access, while `PipelineIntegrationManager` connects the image loader, preprocessing stage, weight initialization, and convolution engine.

The automated tests verify two important properties:

1. Invalid tensor coordinates are detected safely.
2. The convolution engine produces the expected arithmetic result for a known test case.

The overall pipeline is therefore:

Load → Preprocess → Initialize Weights → Convolve → Report Results

This design makes it easier for separate CNN modules to be developed independently while still allowing them to operate together as one integrated system.

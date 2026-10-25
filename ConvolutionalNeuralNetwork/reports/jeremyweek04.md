                ATUGONZA JEREMY – GROUP 10

             Personal Progressive Weekly Report

                     Week 04 Report

Task:

Flattening Layer Un-flatten Gradient Routing Logic.

Work done:

1. Worked on the implementation of the backward operation for the flattening layer in the CNN project.
2. Created the header and source files, flatten_backward.hpp and flatten_backward.cpp.
3. Implemented a function to restore flattened gradients from a one-dimensional vector to the original four-dimensional tensor shape.
4. Used the existing Tensor structure to preserve the original dimensions (N, C, H, W).
5. Implemented validation to ensure that the flattened gradient contains the correct number of values and that the original input is a four-dimensional tensor.
6. Created the test file test_flatten_backward.cpp to check gradient restoration, tensor dimensions, preservation of gradient values, and invalid input handling.
7. Worked on integrating the flattening backward module with the existing C++ project structure.

AI usage:

I used ChatGPT to understand flattening layer backpropagation, gradient routing, tensor dimensions, and how to implement and test the un-flatten operation in C++.

Challenges:

Understanding how to route gradients from the fully connected layer back to the original four-dimensional tensor while preserving the correct order of values. These challenges were addressed by reviewing the tensor structure, implementing the gradient restoration function, and preparing tests for the output dimensions and gradient values.

Conclusion:

The Week 4 work focused on implementing the backward operation for the flattening layer. This operation restores gradients to the original tensor shape, allowing gradients to be passed back to earlier layers of the CNN during training.
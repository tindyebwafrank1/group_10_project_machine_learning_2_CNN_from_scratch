            TINDYEBWA FRANK - GROUP 10 LEADER

            Personal Progressive Weekly Report

                    Week 03 Report

Flattening feature maps (4D to 2D vector conversion).

Work done:

1. Created the flatten.hpp header file in the include folder to declare the flattening function.
2. Created the flatten.cpp file in the src folder to implement the flattening process.
3. Implemented a function that converts a 4D tensor with dimensions (N, C, H, W) into a 2D matrix with dimensions (N, C × H × W).
4. Used the existing Tensor class from the project to access and process tensor elements.
5. Implemented the flattening process while preserving the order of channels, rows, and columns.
6. Created test_flatten.cpp in the tests folder to test the flattening function.
7. Tested the output dimensions, element ordering, and separation of samples in a batch.

AI usage: 
I used ChatGPT to understand how flattening works in a convolutional neural network and how to implement the conversion of a 4D tensor into a 2D matrix. I also used it to understand C++ function declarations, tensor indexing, and the writing of test cases.

Challenges: 
One challenge encountered was a namespace error because the existing Tensor class was declared in the global namespace instead of the `cnn` namespace. This caused compilation errors when using cnn::Tensor. The code was adjusted to use the existing Tensor class correctly.

Conclusion:
The flattening module was developed to convert CNN feature maps into feature vectors for use by the fully connected layer. Testing was prepared to verify the output dimensions, element ordering, and batch separation.
                Atugonza Jeremy
                
        Personal Progressive Weekly Report

                Week 02 Report

Task assigned: 

ReLU Activation Function Module.

Work done:

1. Created relu.hpp in the include folder to declare the ReLU activation functions.
2. Created relu.cpp in the src folder to implement the ReLU activation function.
3. Implemented the ReLU function using the formula f(x)=max(0,x), which replaces negative values with zero while preserving positive values.
4. Implemented the relu() function to apply ReLU to a tensor and return a new tensor containing the activated values.
5. Implemented the reluInPlace() function to apply ReLU directly to an existing tensor.
6. Integrated the ReLU module with the existing Tensor class to allow activation of tensor elements.
7. Created test_relu.cpp in the tests folder to test the activation functions.
8. Tested the module using negative, positive and zero input values to verify that the function produces the expected outputs.
9. Checked that the ReLU function preserves the tensor's shape and does not modify the original tensor when returning a new tensor.
10. Prepared the module for integration with the convolution layer in the CNN pipeline.

AI Usage

I used ChatGPT to understand the ReLU activation function, its mathematical formula, and how to implement it in C++. I also used it to understand tensor operations and develop test cases for verifying the functionality of the module.

Challenges

The main challenge was understanding how to apply the ReLU function to every element of a multidimensional tensor while maintaining its structure. This was addressed by iterating through the tensor's stored values and applying the ReLU formula to each element.

Conclusion

The ReLU activation function module was developed to introduce non-linearity into the convolutional neural network. The module supports both creating a new activated tensor and modifying an existing tensor. It is intended to work with the Tensor class and can be integrated into the CNN pipeline after convolution operations. Testing and integration should be confirmed by running the test program successfully.

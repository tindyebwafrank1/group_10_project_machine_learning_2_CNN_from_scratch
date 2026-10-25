EDRINE – GROUP 10

# Personal Progressive Weekly Report

## Week 01 Report

Task Assigned: Convolution Pipeline Integration + Week-1 Demo.

### Work Done

1. Reviewed the existing convolution interface and identified the cnn::conv2d_single() function used to perform valid 2D convolution.
2. Prepared a demonstration program to connect the tensor input, convolution kernel, and convolution operation.
3. Created a sample single-channel input tensor with dimensions \1\times1\times4\times4).
4. Prepared a (2\times2) convolution kernel and passed it to the existing convolution implementation.
5. Displayed the input tensor, kernel, output feature map, and output dimensions.
6. Added numerical checks to verify selected feature-map values and confirm the expected output dimensions.
7. Prepared an automated test program to check convolution dimensions, numerical results, and invalid-input handling.

### Testing and Verification

The demonstration and automated tests are intended to verify that a (4\times4) input and a (2\times2) kernel produce a (3\times3) output feature map.

**Test outcome:** To be completed after compiling and running the programs.

### Challenges

The integration depends on the existing Tensor class and convolution implementation. Their source files must be linked correctly during compilation.

### Conclusion

The Week 1 deliverables establish a basic convolution demonstration and a test program for verifying the existing convolution module. This provides a foundation for integrating additional CNN operations in later project stages.

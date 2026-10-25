EDRINE – GROUP 10

# Personal Progressive Weekly Report

## Week 02 Report

Task Assigned: CNN Feature Integration + Week-2 Demo.

### Work Done

1. Reviewed the existing convolution, ReLU activation, tensor, and max-pooling modules to prepare them for integration.
2. Created a Week 2 demonstration program to connect the existing CNN operations into a sequential feature-extraction pipeline.
3. Implemented the integration of the input tensor with the convolution operation using the existing cnn::conv2d_single() function.
4. Integrated ReLU activation using the existing cnn::relu() function to remove negative values from the convolution feature map.
5. Integrated max pooling using the existing maxPool2D() function to reduce the spatial dimensions of the feature map.
6. Displayed the feature maps and tensor dimensions at each stage of the pipeline.
7. Created an automated integration test to check the output dimensions and numerical values produced by convolution, ReLU, and max pooling.
8. Prepared compilation and execution commands for testing the integrated modules.

### Testing and Verification

The demo is designed to process a single-channel input tensor through convolution, ReLU activation, and max pooling.

The expected tensor dimensions are:

- Input: (1, 1, 4, 4)
- After convolution: (1, 1, 3, 3)
- After ReLU: (1, 1, 3, 3)
- After max pooling: (1, 1, 2, 2)

Test outcome: To be updated after successfully compiling and running the demo and automated integration tests.

### Challenges

The main integration challenge is ensuring that the existing modules use compatible Tensor definitions, namespaces, function declarations, and source files during compilation.

### Conclusion

The Week 2 work establishes a demonstration of the CNN feature-extraction pipeline by integrating convolution, ReLU activation, and max pooling. The integration test provides a means of verifying the numerical results and tensor dimensions at each stage. This work supports the continued development of the group's CNN-from-scratch project.
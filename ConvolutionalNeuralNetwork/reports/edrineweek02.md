 CNN Feature Block Pipeline

 Overview
This C++ program demonstrates a simplified Convolutional Neural Network (CNN) feature-processing pipeline. It passes a 4D tensor through a convolution layer, a ReLU activation, and a pooling layer, then prints the final output dimensions.

 Main Components
Tensor: Stores image data in a one-dimensional vector<double> while keeping its logical shape as {batch, channels, height, width}.Its indexing operator maps 4D coordinates to a single array position.
Conv2D: Calculates the output height and width using the kernel size, stride, and padding. It creates an output tensor with the requested number of channels. The data operation is simulated, it is not a real convolution.

Pooling: Reduces the height and width by approximately half to represent downsampling. It copies values in a simplified way rather than performing a true max-pooling or average-pooling operation.

FeatureBlock: Connects the layers in sequence: convolution, ReLU, then pooling. ReLU replaces negative values with zero.
main(): Creates an input tensor with shape 1 × 3 × 14 × 14, configures a feature block, runs the input through it, and prints the output spatial dimensions.

 Program Flow

1. Create a mock input tensor with 1 batch, 3 channels, and a height and width of 14.
2. Apply the simulated convolution using 6 output channels, a 3 × 3 kernel, stride 1,and padding 1.
3. Apply ReLU activation.
4. Apply pooling to reduce the spatial dimensions.
5. Print the processed output height and width.

Expected Output

=== Pipeline Integration Execution ===
Processed Output Spatial Block: 7x7




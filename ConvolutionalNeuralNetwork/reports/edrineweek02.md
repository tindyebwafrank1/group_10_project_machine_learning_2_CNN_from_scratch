

Introduction
This program demonstrates a simple Convolutional Neural Network (CNN) feature pipeline in 
C++. It uses classes to store image data, simulate convolution, reduce the size of feature maps through pooling, and connect the operations into one processing block.

The convolution and pooling operations are simplified simulations. They demonstrate how the classes work together, but they do not perform the full mathematical operations used in a real CNN.

Classes used in the Program
The program contains four classes, these include : Tensor, Conv2D, Pooling, and FeatureBlock. Each class has a specific responsibility in the pipeline.

 The Tensor Class
 This class was created to store image or feature-map data using four dimensions namely batch size, number of channels, height, and width. This format is useful for representing data processed by a CNN.

The data does it holds
This  class has two public data members. vector<double> data which stores all the tensor values in a one-dimensional vector and vector<size_t> shape which stores the dimensions in the order {Batch, Channels, Height, Width}.
These members are public so that the other classes can directly read the dimensions and access the stored values.

Functions that operate on that data
The public constructor Tensor(size_t n, size_t c, size_t h, size_t w) initializes the tensor dimensions and creates enough storage for all its values. Every value is initially set to 0.0 .
The public overloaded function operator()(size_t n, size_t c, size_t h, size_t w) which allows a value to be accessed using four coordinates. It converts those coordinates into one index in the one-dimensional data vector.

 Why are the members public or private, and how is the class used?
The class has no explicitly declared private members. Its data and functions are public to make it straightforward for the other classes to access tensor values and dimensions. In a larger application, the data could be made private and accessed through public functions to provide better protection.

In main(), the program creates an input tensor using Tensor img_input(1, 3, 14, 14). This represents one sample with three channels and a height and width of 14. The other classes create and return additional tensors as data moves through the pipeline.

 The Conv2D Class
This class was created to represent a two-dimensional convolution layer. It stores the layer configuration and creates an output tensor with calculated dimensions.

The data does it holds
This class has five private data members, namely : in_c stores the intended number of input channels, out_c stores the number of output channels, k_size stores the kernel size, stride stores the step used when moving the kernel, and padding stores the amount of padding around the input.
These members are private so that code outside the class cannot directly change the layer's configuration. The class manages these values internally.

The functions that operate on that data
The public default constructor Conv2D() initializes all five configuration values to zero. The public parameterized constructor Conv2D(size_t ic, size_t oc, size_t k, size_t s, size_t p) initializes them with values supplied when the object is created.

The public function Tensor forward(const Tensor& input) calculates the output height and width, creates an output tensor, fills it with simulated values, and returns it. The dimensions are calculated using:
h_out = ((input.shape[2] - k_size + 2 * padding) / stride) + 1;
w_out = ((input.shape[3] - k_size + 2 * padding) / stride) + 1;

 Why are the members public or private, and how is the class used?
The configuration values are private to protect the layer's internal settings. The constructors and forward() function are public so that other classes can create the layer and process data through it.

The FeatureBlock constructor creates the convolution object using the supplied configuration values. When the feature block's `forward()` function runs, it calls `conv.forward(input)` to obtain the convolution output.

 The Pooling Class
Rhis class was created to represent a pooling stage that reduces the height and width of a feature map.

The data does it hold
The class has no explicitly declared data members. Its current implementation does not need configuration values because it always reduces the spatial dimensions by approximately half.

The functions operate on the data?
The public function Tensor forward(const Tensor& input) calculates the output height and width, creates an output tensor, copies selected input values cyclically, and returns the result.
The output dimensions are calculated by dividing the input height and width by two:
h_out = input.shape[2] / 2;
w_out = input.shape[3] / 2;

 Why are the members public or private, and how is the class used?
There are no declared private members in this class. The forward() function is public so that another class can call it.

 The FeatureBlock Class
This class was created  to connect the main stages of the feature pipeline. It combines convolution, ReLU activation, and pooling so that the rest of the program can run these stages through one function call.The FeatureBlock class contains a Pooling object named pool. After convolution and ReLU activation, it calls `pool.forward(conv_out)` to produce the final output tensor.

The data does it hold
The class has two public data members. Conv2D conv which holds the convolution layer, while Pooling pool holds the pooling layer.
They are public in this implementation, so they can be accessed directly from outside the class. In a larger program, they could be private because users normally only need to call the feature block's public `forward()` function.

The  functions operate on that data
The public constructor FeatureBlock(size_t in_c, size_t out_c, size_t k_size, size_t stride, size_t pad) initializes the convolution layer with the supplied settings and creates the pooling object.
The public function Tensor forward(const Tensor& input) coordinates the pipeline. First, it passes the input tensor to conv.forward(input). Next, it checks the convolution output values and changes any negative value to 0.0, applying the ReLU activation. Finally, it passes the result to pool.forward(conv_out) and returns the pooled tensor.

 Why are the members public or private, and how is the class used?
The class does not declare any private members. Its layer objects and functions are public to keep the demonstration simple. Making conv and pool private would improve encapsulation, while keeping the constructor and forward()public would allow the class to be configured and used by main().
In main(), the program creates a feature block using FeatureBlock blocks(3, 6, 3, 1, 1). This specifies 3 input channels, 6 output channels, a kernel size of 3, a stride of 1, and padding of 1. The program then calls blocks.forward(img_input)` and stores the returned tensor in result.

 How the Classes Work Together
The pipeline follows this sequence:
1. Tensor stores the input data and its dimensions.
2. Conv2D calculates new dimensions and simulates convolution.
3. ReLU changes negative values to zero.
4. Pooling reduces the height and width.
5. FeatureBlock coordinates these operations and returns the final tensor.

This design separates the program into classes with distinct responsibilities. FeatureBlock connects them, while Tensor provides the data structure shared by the processing stages.

 Program Execution and Expected Output
The input tensor has a batch size of 1, 3 channels, a height of 14, and a width of 14.
The convolution layer uses a kernel size of 3, stride of 1, and padding of 1. Its output height and width are both 14 because:
((14 - 3 + 2 × 1) / 1) + 1 = 14

Pooling then halves each spatial dimension, producing a height and width of 7. The final tensor dimensions are {1, 6, 7, 7}: one sample, six channels, and a spatial size of 7 by 7.
The expected console output is:

=== Pipeline Integration Execution ===
Processed Output Spatial Block: 7x7
Because the input tensor is initialized with zeros, the output values will also be zero. The program checks the pipeline structure and output dimensions; it does not yet implement the full mathematics of convolution or pooling.

 Conclusion
This program demonstrates how C++ classes can organize a simple CNN-style feature pipeline. Tensor stores data and dimensions, Conv2D simulates convolution, Pooling reduces spatial dimensions, and `FeatureBlock` connects the stages with ReLU activation.
The Conv2D configuration members are private to protect the layer settings, while its constructors and processing function are public. The other classes use public members and functions for simplicity. Overall, the program illustrates constructors, data members, member functions, access control, and cooperation between classes.
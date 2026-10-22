Report on the CNN Forward Pipeline 
1. Introduction
This program implements a simple Convolutional Neural Network (CNN) forward-inference pipeline in C++. It takes a batch of feature maps, converts them into flat vectors, calculates class scores using a fully connected layer, and applies the Softmax function to produce class probabilities.


2. Objectives
The main objectives of the program are to:
Represent batches of feature maps using a four-dimensional tensor.
Flatten each sample's feature maps into a one-dimensional feature vector.
Calculate scores for a fixed number of output classes.
Convert the scores into probabilities using Softmax.
Assemble the processing stages into one pipeline.
Demonstrate and assess the use of encapsulation, inheritance, and polymorphism.

3. Program Structure
The program is organized into the following components:

Tensor — stores numerical data and its dimensions.
FlattenLayer — reshapes feature data into flat vectors.
FullyConnectedLayer — calculates a score for each output class.
SoftmaxActivation — converts class scores into probabilities.
CNNForwardPipeline — coordinates the layers in the correct order.
main() — creates sample input, runs inference, and checks the output shape.
The program uses standard C++ libraries: <vector> for storing data, <cmath> for the exponential function, <iostream> for console output, and <algorithm> for the fill() function.

4. Explanation of the Main Components
 Tensor structure
The Tensor structure stores its values in a one-dimensional vector<double> named data. Its shape vector records four dimensions in this order:

Batch size (N)
Number of channels (C)
Height (H)
Width (W)
Although the data represents a four-dimensional structure, it is stored in contiguous one-dimensional memory. The indexing operator maps a coordinate (n, c, h, w) to a position in that array:

n * (C * H * W) + c * (H * W) + h * W + w

The non-constant operator() returns a reference that allows a value to be modified. The const overload allows values to be accessed when the tensor is constant without permitting modification through that access.

The constructor allocates N × C × H × W elements and initializes them to zero.

 FlattenLayer
The flatten layer changes the shape of the feature data so that each sample can be processed by the fully connected layer.
It calculates the flattened dimension as:
flat_dim = channels × height × width
It then creates an output tensor with shape {batch, 1, 1, flat_dim} and copies the input data into it.

Flattening changes the way the data is represented; it does not perform a learned mathematical transformation on the feature values.

FullyConnectedLayer
The fully connected layer calculates one score for each output class. It stores:
in_dim — the number of input features.
out_dim — the number of output classes.
weights — the connection weights.
biases — the bias for each output class.
The constructor initializes every weight to 0.01 and every bias to 0.0. These are fixed initial values; the program does not train the weights.

For batch sample b and output class o, the score is calculated as:

score = bias[o] + sum(input[b, i] * weight[i, o])

The loops process every sample, every output class, and every input feature. The resulting scores are stored in a tensor with shape {batch, 1, 1, out_dim}.

SoftmaxActivation
The Softmax layer converts raw class scores, also called logits, into probabilities.

For each sample, it first finds the maximum score and subtracts it from each score before calculating exponentials. This is a common numerical-stability technique that helps prevent exponential overflow.

For class i, Softmax is:

P(i) = exp(score[i] - max_score) / sum(exp(score[j] - max_score))

The resulting probabilities for each sample sum to approximately 1, subject to floating-point rounding. A larger probability indicates a class that the current model scores more strongly.

The input is expected to contain at least one class per sample. The current code does not explicitly validate this condition.

 CNNForwardPipeline
The CNNForwardPipeline class combines the three processing stages:

Flatten the input features.
Pass the flat vectors through the fully connected layer.
Apply Softmax to the raw scores.
Its execute_inference() method returns the final probability tensor. This design keeps the sequence of operations in one place and makes the overall inference process easier to use.

The pipeline's constructor initializes the fully connected layer with the supplied flattened input dimension and number of classes.

The main function
The main() function sets the following values:

Batch size: 2
Channels: 8
Height: 16
Width: 16
Number of target classes: 10
The flattened feature dimension is:

8 × 16 × 16 = 2048

The program creates an input tensor with shape {2, 8, 16, 16} and fills all its values with 0.25. It then executes the pipeline.

The expected output shape is:

{2, 1, 1, 10}

This means the program produces ten class probabilities for each of the two input samples. If the final dimension is ten, the program prints Forward assembly pipeline verification: PASSED. Otherwise, it prints FAILED.

This check verifies the output's class dimension only. It does not prove that the probabilities are correct or that the model has learned to classify real data.

 Object-Oriented Programming Concepts
 Encapsulation
Encapsulation is partially demonstrated in this program. It involves keeping data and the operations that work on that data together, while controlling access to internal details.

The clearest example is FullyConnectedLayer, which declares in_dim, out_dim, weights, and biases under private:. Code outside the class cannot directly access these members. The public constructor and forward() method provide the intended way to create the layer and calculate its output.

This protects the layer's internal representation from arbitrary direct modification by other parts of the program.

The Tensor, FlattenLayer, and SoftmaxActivation components are declared as structs. In C++, members of a struct are public by default, unless an access label says otherwise. Therefore, their data and methods are generally accessible from outside. The program uses encapsulation in FullyConnectedLayer, but it does not apply strict data hiding consistently to every component.

 Inheritance
Inheritance is not implemented in the supplied code.
Inheritance allows a new class to derive from an existing class and reuse or extend its members. For example, a design could introduce a common base class called Layer, then derive FlattenLayer, FullyConnectedLayer, and SoftmaxActivation from it.

The supplied code does not declare a base class or use C++ inheritance syntax such as:

class FullyConnectedLayer : public Layer

Each component is defined independently. The CNNForwardPipeline class contains layer objects as members; this is an example of composition, not inheritance. Composition means that one class is built using objects of other classes.

A shared base class could be added if the project requires inheritance, but it is not necessary for this simple pipeline to work.

5.3 Polymorphism
Runtime polymorphism is not implemented in the supplied code.

Polymorphism allows the same interface to produce different behavior depending on the object being used. In C++, runtime polymorphism is commonly implemented through virtual functions and base-class pointers or references.

For example, if the program had a base class Layer with a virtual forward() function, derived layer classes could override that function. The pipeline could then call forward() through a Layer reference or pointer and execute the correct implementation for the actual layer type.

In the current code, FlattenLayer::forward(), FullyConnectedLayer::forward(), and SoftmaxActivation::forward() are separate functions. They have similar names but are not overrides of a shared virtual function. The pipeline calls each concrete object directly. This is ordinary member-function calling, not runtime polymorphism.

 Summary of the three concepts
Encapsulation: Present in FullyConnectedLayer through its private data members and public operations, though not consistently applied across all components.

Inheritance: Not present. No class derives from another class.
Polymorphism: Runtime polymorphism is not present. There is no shared base interface with virtual overridden functions.
Composition: Present. CNNForwardPipeline contains objects of FlattenLayer, FullyConnectedLayer, and SoftmaxActivation.

 Execution Flow
The inference process follows this sequence:

main() creates the input tensor.
The input values are set to 0.25.
CNNForwardPipeline::execute_inference() receives the input features.
FlattenLayer converts each sample's feature maps into a flat vector.
FullyConnectedLayer computes ten raw class scores for each sample.
SoftmaxActivation converts each set of scores into probabilities.
main() prints the output dimensions and checks whether the number of classes is ten.
The data flow can be summarized as:

Input Tensor → Flatten → Fully Connected → Softmax → Output Probabilities

 Limitations and Possible Improvements
No training process: Weights and biases are initialized to fixed values. There is no loss calculation, backpropagation, or optimization.
No convolution layer: Despite the CNN name, this program starts with precomputed feature maps and implements only flattening, a fully connected layer, and Softmax.
No dimension validation: The fully connected layer assumes that the input feature count matches in_dim. Incorrect dimensions could cause invalid memory access.
No empty-class validation: Softmax assumes at least one class. An input with zero classes would make its access to the first score invalid.
Limited verification: The test checks only results.shape[3]. It could also check the complete output shape, confirm that probabilities are finite, and check that each sample's probabilities sum approximately to one.
Basic encapsulation: Tensor exposes its storage publicly. A more defensive design could hide its internal data and validate indices.
No inheritance or runtime polymorphism: If a common layer interface is a project requirement, a base class with a virtual forward() method and derived layer classes could be introduced.
Fixed initialization: Initializing all weights to the same value is suitable for a demonstration but is not a good general strategy for training a neural network.

                                 Atugonza Jeremy
                    Personal Progressive Weekly Report

    Week 03 Report

Task assigned: Weight and Bias Initialization using Xavier and He Methods.

Work done:

1. Created the weight initialization module to support the initialization of weights used in the fully connected layers of the Convolutional Neural Network (CNN).
2. Defined the InitializationMethod enumeration with two methods: Xavier and He initialization.
3. Implemented the initializeWeights() function to generate weight values based on the selected initialization method.
4. Implemented Xavier uniform initialization using the limit calculated from the number of input and output neurons.
5. Implemented He normal initialization using a standard deviation calculated from the number of input neurons.
6. Used C++ random number generation facilities, including random_device, mt19937, and probability distributions, to generate the initial weight values.
7. Added input validation to ensure that fan_in and fan_out are greater than zero before initializing weights.
8. Implemented the initializeBiases() function to initialize all bias values to zero.
9. Created a test file, test_weight_initialization.cpp, to verify the number of generated weights, the Xavier initialization limits, the validity of He initialization values, and zero-initialized biases.
10. Added a test for invalid dimensions to verify that the initialization function throws an exception when the number of input neurons is zero.

Object-Oriented Programming Concepts

1. Inheritance

Inheritance allows a derived class to acquire accessible properties and behaviours from a base class.

In our CNN project, inheritance can be used to create specialized neural-network layers from a common base layer. However, inheritance should only be reported as implemented if the project actually contains a base class and derived classes.

For example in the pooliing.hpp header, Pooling2D is the base class while MaxPooling2D and AveragePooling2D are the derived classes.

2. Encapsulation

Encapsulation involves combining data and functions within a class and controlling access to the internal data using access specifiers.

In our project, the Tensor class represents multidimensional numerical data using its shape and stored values. Its member functions provide operations for accessing and manipulating the data.

The Image class also demonstrates encapsulation by storing image dimensions and pixel information and providing functions for accessing pixels.

Importance: Encapsulation helps organize the code, protect internal data from uncontrolled access, and make the project easier to maintain.

3. Polymorphism

Polymorphism allows a common interface to represent different implementations of an operation. In C++, it can be implemented through function overriding using virtual functions or through function overloading.

For example, a CNN could define a common forward() interface for convolutional, pooling, and fully connected layers, with each layer implementing its own operation.

Class Layout Drawings

As instructed, I have attached a clear photograph of the hand drawn class layouts to this report.

The drawings show:
The Tensor class, including its data members, member functions, and access specifiers.
The Image class, including its image data and pixel-access functions.
Any actual inheritance relationships present in the project.
Any actual polymorphic functions and their overriding classes, if present.
The class diagrams will be based on the project's source code to ensure that the member names, access specifiers, and relationships are accurate.

AI usage:

I used ChatGPT to understand the purpose of weight and bias initialization in neural networks and how Xavier and He initialization methods work. ChatGPT also helped me understand the implementation of random number generation, input validation, and unit tests in C++.

Challenges:

The main challenge was understanding the mathematical differences between Xavier and He initialization and how to translate their formulas into C++ code. Another challenge was ensuring that the generated weights satisfy the expected mathematical constraints and that invalid input dimensions are handled correctly. These challenges were addressed by reviewing the formulas, implementing the appropriate probability distributions, and writing unit tests to check the initialization functions.

Conclusion:

The weight and bias initialization component was developed to support the fully connected layers of the CNN. The module provides Xavier uniform and He normal weight initialization, together with zero-initialized biases. Tests were prepared to verify the functionality, expected weight ranges, bias values, and handling of invalid dimensions. This component provides a foundation for the fully connected classification stage and the subsequent training and optimization processes.
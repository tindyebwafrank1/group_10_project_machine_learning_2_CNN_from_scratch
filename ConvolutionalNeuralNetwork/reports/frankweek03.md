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
I used ChatGPT to understand how flattening works in a convolutional neural network and how to implement the conversion of a 4D tensor into a 2D matrix. I also used it to understand C++ function declarations, tensor indexing, and the writing of test cases.

Challenges: 
One challenge encountered was a namespace error because the existing Tensor class was declared in the global namespace instead of the `cnn` namespace. This caused compilation errors when using cnn::Tensor. The code was adjusted to use the existing Tensor class correctly.

Conclusion:
The flattening module was developed to convert CNN feature maps into feature vectors for use by the fully connected layer. Testing was prepared to verify the output dimensions, element ordering, and batch separation.

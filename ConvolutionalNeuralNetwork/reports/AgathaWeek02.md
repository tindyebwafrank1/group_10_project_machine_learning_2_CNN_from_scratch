# Week 2 Report – Classes in the Program

**Name:** Agatha

## Classes Used

The program is organized using different classes. The main classes include:

- Tensor
- Sigmoid
- Tanh

## Tensor Class

### Why was it created?

The Tensor class stores the data used throughout the program.

### Data Members

- N
- C
- H
- W
- Data vector

### Member Functions

- Constructor
- size()
- fill()
- reshape()
- add()
- subtract()
- multiply()
- scale()
- operator()

### Access Specifiers

**Private**
- Tensor dimensions
- Data vector

**Public**
- Constructor
- Member functions

## Sigmoid Class

### Why was it created?

To apply the sigmoid activation function.

### Member Functions

- forward()
- backward()

## Tanh Class

### Why was it created?

To apply the tanh activation function.

### Member Functions

- forward()
- backward()

## Class Design

The Tensor class stores data, while the activation classes perform calculations on that data. This makes the program easier to understand and maintain.

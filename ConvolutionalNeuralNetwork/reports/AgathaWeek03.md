# Week 3 Report – Encapsulation, Inheritance and Polymorphism

**Name:** Agatha

## Encapsulation

The program uses encapsulation by keeping important data inside classes.

For example, the Tensor class keeps its data members private while public member functions are used to access and modify the data.

## Inheritance

The current program does not directly use inheritance.

If needed, a parent Activation class could be created, with Sigmoid and Tanh inheriting from it.

## Polymorphism

The program does not use runtime polymorphism because there are no virtual functions or base classes.

Each activation class has its own forward() and backward() functions.

## Class Layout

### Tensor

**Private**
- N
- C
- H
- W
- Data vector

**Public**
- Constructor
- size()
- fill()
- reshape()
- add()
- subtract()
- multiply()
- scale()
- operator()

### Sigmoid

**Public**
- forward()
- backward()

### Tanh

**Public**
- forward()
- backward()

## Conclusion

The program mainly demonstrates encapsulation through the Tensor class. Inheritance and polymorphism are not implemented, but the classes are organized in a way that allows them to be added in future versions.

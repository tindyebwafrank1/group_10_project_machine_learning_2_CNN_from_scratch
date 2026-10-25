# Reusable Convolutional Neural Network (CNN) Library in C++

A modular C++17 library project for implementing the core operations of
a convolutional neural network (CNN), developed as the Machine Learning
2 (ML2) group project.

> **Project status:** In development\
> **Language standard:** C++17\
> **Build system:** CMake\
> **Project deadline:** 5 November 2026

## Overview

This project aims to build a reusable CNN library in C++ with clearly
separated public interfaces, implementations, tests, and example
programs. The library is being developed module by module, with unit
tests and integration checks used to verify numerical operations, tensor
dimensions, and data flow through the network.

The intended workflow covers image loading and preprocessing, tensor
operations, convolution, padding and strides, activation functions,
pooling, dense layers, probability and loss calculations, and
prediction/evaluation utilities.

This repository is an academic engineering project. Its implementation
and test coverage should be considered in progress until the relevant
modules have been completed and verified.

## Project Objectives

-   Provide reusable C++ components for core CNN operations.
-   Represent and manipulate tensors with dynamic dimensions.
-   Implement 2D convolution, padding, stride handling, and spatial
    pooling.
-   Provide common activation functions, including ReLU, Sigmoid, and
    Leaky ReLU.
-   Support dense-layer calculations, Softmax, and cross-entropy loss.
-   Connect the components through a CNN pipeline.
-   Check tensor shapes and invalid inputs at module boundaries.
-   Maintain a modular codebase with unit tests and runnable examples.

## Planned Features

  -----------------------------------------------------------------------
  Component                           Intended responsibility
  ----------------------------------- -----------------------------------
  Image loading and preprocessing     Load image data, validate paths and
                                      inputs, and scale/normalize pixel
                                      values

  Tensor operations                   Dynamic N-dimensional
                                      representation, indexing, shape
                                      checks, and numerical safety

  2D convolution                      Apply convolution kernels and
                                      accumulate feature values

  Padding and strides                 Handle spatial borders and control
                                      filter movement

  Activation functions                Element-wise ReLU, Sigmoid, and
                                      Leaky ReLU

  Pooling                             Max pooling and average pooling
                                      over spatial windows

  Dense layer                         Flatten features and perform
                                      weight-based matrix/vector
                                      calculations

  Softmax and cross-entropy           Convert scores to probabilities and
                                      calculate classification loss

  Pipeline and verification           Connect processing stages and
                                      verify dimension compatibility

  Evaluation utilities                Support prediction and evaluation
                                      measures as implementation
                                      progresses
  -----------------------------------------------------------------------

These are the planned responsibilities from the project roadmap;
availability and completeness depend on the current state of the source
code.

## Repository Structure

The intended repository layout is:

``` text
.
├── CMakeLists.txt
├── include/
│   └── cnn/
│       ├── cnn_pipeline.hpp
│       ├── image_loader.hpp
│       ├── tensor.hpp
│       ├── conv2d.hpp
│       ├── padding.hpp
│       ├── activations.hpp
│       ├── pooling.hpp
│       ├── dense_loss.hpp
│       └── routing_verifier.hpp
├── src/
│   ├── cnn_pipeline.cpp
│   ├── image_loader.cpp
│   ├── tensor.cpp
│   ├── conv2d.cpp
│   ├── pooling.cpp
│   └── dense_loss.cpp
├── tests/
│   ├── test_image_loader.cpp
│   ├── test_tensor.cpp
│   ├── test_conv2d.cpp
│   ├── test_pooling.cpp
│   ├── test_dense_loss.cpp
│   └── test_routing.cpp
├── examples/
│   └── cnn_demo.cpp
└── reports/
    ├── week-01.md
    ├── week-02.md
    ├── week-03.md
    └── final-report.md
```

The tree illustrates the planned organization. Actual filenames may
differ as the implementation evolves. Keep public declarations in
`include/`, implementation files in `src/`, automated checks in
`tests/`, and runnable demonstrations in `examples/`.

## Requirements

Before building, ensure the development environment has:

-   A C++ compiler with C++17 support (for example, a recent GCC, Clang,
    or MSVC toolchain).
-   CMake 3.15 or newer.
-   Git.
-   A test framework or test setup if required by the repository's
    current CMake configuration.

Any additional image-decoding library or third-party dependency must be
documented when it is selected and added to the build.

Tests should cover normal calculations and edge cases, including invalid
file paths, incompatible tensor shapes, boundary indexing, convolution
reference matrices, stride/padding combinations, pooling windows, and
numerical stability.

## Run the Example

The planned demonstration entry point is `examples/cnn_demo.cpp`. Once
the example is included as a CMake executable target, build it using the
target name configured in CMake and run the resulting executable.

The exact command depends on the target name and the platform. A
runnable end-to-end demo should be documented here after it has been
verified in a clean build.

## Design and Engineering Principles

-   **Modularity:** Keep CNN operations separated into focused
    components.
-   **Interface/implementation separation:** Place public declarations
    in headers and implementation logic in source files.
-   **Validation:** Check dimensions, parameters, and input paths at
    module boundaries.
-   **Numerical care:** Test edge cases and guard against invalid
    numerical values where appropriate.
-   **Testability:** Use small, hand-calculated reference cases
    alongside integration tests.
-   **Maintainability:** Prefer clear names, consistent formatting, and
    focused changes.
-   **Review before merge:** Changes should be reviewed and tested
    before integration.

## Contribution Workflow

The project roadmap establishes the following Git and pull-request
rules:

1.  Create or use an assigned feature branch, for example
    `feature/conv2d`.
2.  Keep changes focused on the assigned module or task.
3.  Build the project and run relevant tests before opening a pull
    request.
4.  Open a pull request against the shared integration branch.
5.  Request review from the assigned reviewer/supervisor.
6.  Address review feedback and wait for approval before merging.
7.  The team lead merges approved pull requests; contributors must not
    merge their own pull requests.
8.  Do not commit directly to `main`.

Use descriptive commit messages and include relevant test results in the
pull request description. Follow the team's current branch-protection
and review settings.

## Quality Assurance

The roadmap calls for:

-   Warning-conscious builds, including `-Wall -Wextra -pedantic` where
    supported.
-   Unit tests for individual modules.
-   Integration tests for tensor shapes and pipeline data flow.
-   Checks for invalid inputs and numerical edge cases.
-   Memory-safety and leak checks, including Valgrind where available
    and appropriate.
-   A clean-environment build before final submission.

These are project quality goals, not a claim that all checks currently
pass. Record actual test results and known limitations as development
proceeds.

## Project Documentation

Weekly engineering reports and the final report are planned under
`reports/`. Reports should describe completed work, verification
performed, known issues, and any AI-assisted tools used, including the
tool, purpose, and technical justification, in accordance with the
course project rules.

## Team and Responsibilities

The project roadmap assigns work across the team:

-   **Frank** --- architecture, pipeline integration, and team
    coordination.
-   **Peter** --- image loading, preprocessing, and Operational Risk
    checks.
-   **Agatha** --- tensor structures, indexing, and numerical safety.
-   **Allan Mugarura** --- 2D convolution operations.
-   **Jeremie** --- padding, strides, and activation functions.
-   **Nats JoJo** --- max and average pooling.
-   **Edrine Mugenyi** --- dense layer, Softmax, cross-entropy, and
    evaluation.
-   **Sempewo Evarita** --- workflow routing, dimension verification,
    and integration checks.

Refer to the team's current task plan for detailed ownership and
reviewer assignments.

## Scope and Limitations

This project focuses on implementing foundational CNN operations in C++.
Do not assume support for model training, automatic differentiation, GPU
acceleration, pretrained networks, or production deployment unless those
capabilities are explicitly implemented and documented.


## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for the full license text.

Copyright (c) 2026  Group 10


## Acknowledgements

Developed as a group project for **Machine Learning 2 (ML2)**. The project
repository and documentation reflect the team's own
implementation, testing, and engineering decisions.

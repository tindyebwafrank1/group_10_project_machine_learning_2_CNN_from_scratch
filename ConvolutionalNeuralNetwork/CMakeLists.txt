cmake_minimum_required(VERSION 3.15)
project(ConvolutionalNeuralNetwork
    VERSION 1.0
    LANGUAGES CXX
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include_directories(include)
add_executable(ConvolutionalNeuralNetwork 
    src/main.cpp
    src/tensor.cpp
)

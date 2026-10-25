cmake_minimum_required(VERSION 3.15)

project(ConvolutionalNeuralNetwork
    VERSION 1.0
    LANGUAGES CXX

)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable warnings for cleaner builds
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Public include directory for all targets
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Core CNN library sources
set(CNN_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cnn_forward_pipeline.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/convolution_backward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cross_entropy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_block.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/feature_integration.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/filter_initialization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flatten.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/flatten_backward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/image_loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/loss_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/max_pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/metrics.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/multi_channel_convolution.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/padding.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pooling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pooling_backward.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/prediction.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/relu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/sgd_optimizer.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/weight_initialization.cpp
)

# Build a reusable library
add_library(cnn_lib STATIC ${CNN_SOURCES})
target_include_directories(cnn_lib
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Main executable
add_executable(ConvolutionalNeuralNetwork
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp
)

target_link_libraries(ConvolutionalNeuralNetwork PRIVATE cnn_lib)
target_include_directories(ConvolutionalNeuralNetwork
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Build all example programs automatically
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
)

foreach(example_src IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name "${example_src}" NAME_WE)
    add_executable(${example_name} "${example_src}")
    target_link_libraries(${example_name} PRIVATE cnn_lib)
    target_include_directories(${example_name}
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
endforeach()

# Build all tests automatically
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests")
    enable_testing()

    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    )

    foreach(test_src IN LISTS TEST_SOURCES)
        get_filename_component(test_name "${test_src}" NAME_WE)
        add_executable(${test_name} "${test_src}")
        target_link_libraries(${test_name} PRIVATE cnn_lib)
        target_include_directories(${test_name}
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
        add_test(NAME ${test_name} COMMAND ${test_name})
    endforeach()
endif()
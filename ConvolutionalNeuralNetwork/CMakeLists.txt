
cmake_minimum_required(VERSION 3.15)

project(ConvolutionalNeuralNetwork
    VERSION 1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Compiler warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Project headers
set(CNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Reusable CNN implementation modules.
# Standalone legacy programs are intentionally excluded.
set(CNN_SOURCES
    src/convolution.cpp
    src/convolution_backward.cpp
    src/cross_entropy.cpp
    src/feature_block.cpp
    src/filter_initialization.cpp
    src/flatten.cpp
    src/flatten_backward.cpp
    src/fully_connected.cpp
    src/fully_connected_backward.cpp
    src/image.cpp
    src/image_loader.cpp
    src/loss_gradient.cpp
    src/max_pooling.cpp
    src/metrics.cpp
    src/multi_channel_convolution.cpp
    src/padding.cpp
    src/pooling.cpp
    src/pooling_backward.cpp
    src/prediction.cpp
    src/relu.cpp
    src/sgd_optimizer.cpp
    src/strides.cpp
    src/tensor.cpp
    src/training.cpp
    src/weight_initialization.cpp
)

# Reusable CNN library
add_library(cnn_lib STATIC ${CNN_SOURCES})

target_include_directories(cnn_lib
    PUBLIC "${CNN_INCLUDE_DIR}"
)

# Build demonstration programs separately.
# Temporarily skip the demo that includes a missing header.
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
)

foreach(example_src IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name "${example_src}" NAME_WE)

   if(example_name STREQUAL "fully_connected_training_demo"
   OR example_name STREQUAL "fully_connected_demo"
   OR example_name STREQUAL "stride_demo")

    message(STATUS "Temporarily skipping ${example_name}: needs fixing")

else()
    add_executable("${example_name}" "${example_src}")
    target_link_libraries("${example_name}" PRIVATE cnn_lib)
endif()
endforeach()

# Build and register unit tests
include(CTest)

if(BUILD_TESTING)
    file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp"
    )

    foreach(test_src IN LISTS TEST_SOURCES)
        get_filename_component(test_name "${test_src}" NAME_WE)

        add_executable("${test_name}" "${test_src}")
        target_link_libraries("${test_name}" PRIVATE cnn_lib)

        
        add_test(
            NAME "${test_name}"
            COMMAND "${test_name}"
        )

        set_tests_properties("${test_name}" PROPERTIES
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        )

    endforeach()
endif()

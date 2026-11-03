cmake_minimum_required(VERSION 3.10)

project(ConvolutionalNeuralNetwork LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Project directories
set(CNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Collect source files
file(GLOB CNN_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# Exclude standalone programs containing their own main()
# and duplicate Tensor implementations.
list(FILTER CNN_SOURCES EXCLUDE REGEX
    ".*/(cnn_forward_pipeline|feature_integration)\\.cpp$"
)

# Reusable CNN library
add_library(cnn_lib STATIC ${CNN_SOURCES})

target_include_directories(cnn_lib
    PUBLIC "${CNN_INCLUDE_DIR}"
)

# Demonstration programs
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
)

foreach(example_src IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name "${example_src}" NAME_WE)

    add_executable("${example_name}" "${example_src}")

    target_link_libraries("${example_name}" PRIVATE cnn_lib)
endforeach()

# Testing
enable_testing()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
)

foreach(test_src IN LISTS TEST_SOURCES)
    get_filename_component(test_name "${test_src}" NAME_WE)

    add_executable("${test_name}" "${test_src}")

    target_link_libraries("${test_name}" PRIVATE cnn_lib)

    add_test(NAME "${test_name}" COMMAND "${test_name}")

    set_tests_properties("${test_name}" PROPERTIES
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
endforeach()

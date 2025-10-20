cmake_minimum_required(VERSION 3.18)
project(cuda_image_filters LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)  # Turing, Ampere, Ada architectures

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Enable relocatable device code for separate compilation
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_SOURCE_DIR}/src)

# Source files
set(CUDA_SOURCES
    src/convolution.cu
    src/gaussian_blur.cu
    src/sobel.cu
    src/box_blur.cu
    src/sharpening.cu
)

# Create library
add_library(cuda_filters SHARED ${CUDA_SOURCES})

# Link CUDA libraries
target_link_libraries(cuda_filters CUDA::cudart)

# Set properties
set_target_properties(cuda_filters PROPERTIES
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
)

# Optional: Build test executable
option(BUILD_TESTS "Build test executables" ON)
if(BUILD_TESTS)
    add_executable(filter_test src/main.cpp)
    target_link_libraries(filter_test cuda_filters)
endif()

# Python bindings
option(BUILD_PYTHON "Build Python bindings" ON)
if(BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
    find_package(pybind11 CONFIG)
    
    if(pybind11_FOUND)
        pybind11_add_module(fastfilters python_bindings/bindings.cpp)
        target_link_libraries(fastfilters PRIVATE cuda_filters)
        target_include_directories(fastfilters PRIVATE ${CMAKE_SOURCE_DIR}/src)
    else()
        message(WARNING "pybind11 not found, skipping Python bindings")
    endif()
endif()

# Installation
install(TARGETS cuda_filters
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY src/
    DESTINATION include/cuda_filters
    FILES_MATCHING PATTERN "*.cuh" PATTERN "*.h"
)

# Print configuration
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")

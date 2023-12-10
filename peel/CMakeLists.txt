cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
# set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
set(CMAKE_CUDA_ARCHITECTURES "86")
project(Learn_Graph_Computing CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(gpp
        shared/graph.h
        shared/graph.cpp
        shared/globals.h
        shared/virtual_graph.cpp
        shared/virtual_graph.h
        shared/argument.cpp
        shared/argument.h
        shared/Timer.cpp
        shared/Timer.h
        shared/util.cpp
        shared/util.h
        shared/gpu_error.cuh
        gppeel.cu)

set_target_properties(
        gpp
        PROPERTIES
        # CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES  86
)

add_executable(peelone
        shared/graph.h
        shared/graph.cpp
        shared/globals.h
        shared/virtual_graph.cpp
        shared/virtual_graph.h
        shared/argument.cpp
        shared/argument.h
        shared/Timer.cpp
        shared/Timer.h
        shared/util.cpp
        shared/util.h
        shared/gpu_error.cuh
        peelone.cu
        )

set_target_properties(
        peelone
        PROPERTIES
        # CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES  86
)
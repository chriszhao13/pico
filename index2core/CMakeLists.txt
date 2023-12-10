cmake_minimum_required(VERSION 3.10)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda-11.7/bin/nvcc")
# set(CMAKE_CXX_COMPILER "/usr/bin/gcc")
set(CMAKE_CUDA_ARCHITECTURES "86")
project(Learn_Graph_Computing CUDA CXX)

set(CMAKE_CXX_STANDARD 14)

add_executable(HistoCore
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
        HistoCore.cu)

set_target_properties(
        HistoCore
        PROPERTIES
        # CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES  86
)

add_executable(NbrCore
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
        NbrCore.cu)

set_target_properties(
        NbrCore
        PROPERTIES
        # CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES  86
)

add_executable(CntCore
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
        CntCore.cu
        )

set_target_properties(
        CntCore
        PROPERTIES
        # CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES  86
)
cmake_minimum_required(VERSION 3.20)
project (project_cuda LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)

add_executable(project_cuda main.cu)

set_target_properties(project_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

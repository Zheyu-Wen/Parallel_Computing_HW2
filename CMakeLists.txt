cmake_minimum_required(VERSION 3.9.0)
project(zheyu_hw2 LANGUAGES CXX CUDA)

SET(CMAKE_CXX_FLAGS "-std=c++11 -lstdc++fs")
SET(CMAKE_CXX_STANDARD_REQUIRED)


SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
SET(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED)
find_package(OpenMP REQUIRED)

include_directories(include
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${MPI_CXX_INCLUDE_DIRS})

set(LIBS
    #CUDA::cudart	
    MPI::MPI_CXX
    OpenMP::OpenMP_CXX)

add_library(hw2 
    src/parscan.cu)

target_compile_features(hw2 PRIVATE cxx_std_11)
set_target_properties(hw2
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(parscan_main src/parscan_main.cpp)
set_property(TARGET parscan_main 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(parscan_main PRIVATE hw2 ${LIBS})

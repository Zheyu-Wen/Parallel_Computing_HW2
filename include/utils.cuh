#ifndef UTILS_CUH
#define UTILS_CUH

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <unistd.h>

using namespace std;
#define tic chrono::high_resolution_clock::now()
#define toc chrono::high_resolution_clock::now()
#define milliseconds(x) std::chrono::duration_cast<std::chrono::milliseconds>(x)
#define seconds(x) std::chrono::duration_cast<std::chrono::seconds>(x)

__global__ void matvec_kernel(float (*L)[114], const boost::array<float, 114>& tau_a, const boost::array<float, 114>& matvec_out);
void matvec_func(float (*L)[114], float* tau_a, float* matvec_out);

#endif // UTILS_CUH

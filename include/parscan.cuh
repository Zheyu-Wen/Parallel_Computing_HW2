#ifndef PARSCAN_CUH
#define PARSCAN_CUH

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

__global__ void sequential_sum(float* a, float* s, int n);
float* seq_scan_cuda(float *a, int n);
void seq_scan_mpicuda(int n, int rank);
void parscan_mpicuda(int n, int rank);
float* parscan_cuda(float *a, int n);
#endif // PARSCAN_CUH

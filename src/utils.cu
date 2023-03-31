#include "utils.cuh"

__global__ void matvec_kernel(float (*L)[114], float* tau_a, float* matvec_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=idx+1; i<n; i+= blockDim.x * gridDim.x){
        matvec_out[i] = 0
        for (int j=0; j<=114; j++){
            matvec_out[i] += L[i][j] * tau_a[j];
        }
    }
}

void matvec_func(float (*L)[114], const boost::array<float, 114>& tau_h, const boost::array<float, 114>& out_h) {
    
    const int threads_per_block = 8;
    int numBlocks = (114 + threads_per_block  - 1) / threads_per_block;
    
    size_t buf_L = 114*114*sizeof(float);
    size_t buf_tau = 114*sizeof(float);
   
    float (*L_d)[114];
    if (cudaMalloc((void**)&(*L_d)[114], buf_L) != cudaSuccess) {
        perror("cuda malloc failed!");
    }
    float* tau_d;
    if (cudaMalloc((void**)&tau_d, buf_tau) != cudaSuccess) {
        perror("cuda malloc failed!");
    }
    float* out_d;
    if (cudaMalloc((void**)&out_d, buf_tau) != cudaSuccess) {
        perror("cuda malloc failed!");
    }
   
    cudaMemcpy(L_d, L_h, buf_L, cudaMemcpyHostToDevice);
    cudaMemcpy(tau_d, tau_h.data(), buf_tau, cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out_h.data(), buf_tau, cudaMemcpyHostToDevice);

    cudaEvent_t start1, end1;
    cudaEventCreate(&start1);
    cudaEventCreate(&end1);
    cudaEventRecord(start1, 0);

    matvec_kernel<<<numBlocks, threads_per_block>>>(L_d, tau_d, out_d);

    cudaEventRecord(end1, 0);
    cudaEventSynchronize(end1);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, end1);

    cudaMemcpy(out_h.data(), out_d, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(L_d);
    cudaFree(tau_d);
    
    printf("cuda run time %3.1f ms\n", rank, milliseconds);
}

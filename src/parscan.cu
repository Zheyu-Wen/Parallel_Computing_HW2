#include "parscan.cuh"

__global__ void sequential_sum(float* a, float* s, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=idx+1; i<n; i+= blockDim.x * gridDim.x){
        for (int j=0; j<=i; j++){
            s[i] += a[j];
        }
    }
}

__global__ void pair_sum(float* a, float* b, int m){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=idx; i<m; i+=blockDim.x * gridDim.x){
        b[i] = a[i*2] + a[i*2+1];
    }
}

__global__ void assign_ans(float* a, float* c, float* s, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=idx; i<n; i+=blockDim.x * gridDim.x){
        if (i%2==1) s[i] = c[(i-1)/2];
	else s[i] = a[i] + c[i/2 - 1];
    }
}

__global__ void sequential_add(float* a, int n, float addition) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i=idx; i<n; i+= blockDim.x * gridDim.x){
        a[i] += addition;
    }
}

void parscan_mpicuda(int n, int rank) {
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    const int threads_per_block = 8;
    int numBlocks = (n + threads_per_block  - 1) / threads_per_block;
    
    int m = n / 2;
    size_t buf_size = n*sizeof(float);
    size_t buf_m = m*sizeof(float);
    float* a = (float*)malloc(buf_size);
    float* b = (float*)malloc(buf_m);
    float* s = (float*)malloc(buf_size);

    MPI_Status mpi_status;
    float* da;
    cudaMalloc((void**)&da, buf_size);
    float* db;
    cudaMalloc((void**)&db, buf_m);
    MPI_Recv(a, n, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &mpi_status);
    cudaMemcpy(da, a, buf_size, cudaMemcpyHostToDevice);
    pair_sum<<<numBlocks, threads_per_block>>>(da, db, m);
    cudaMemcpy(b, db, buf_m, cudaMemcpyDeviceToHost);
    
    //float* c = parscan_cuda(b, m);

    float* dc;
    cudaMalloc((void**)&dc, buf_m);
    sequential_sum<<<numBlocks, threads_per_block>>>(db, dc, n);
    //cudaMemcpy(dc, c, buf_m, cudaMemcpyHostToDevice);

    float* ds;
    cudaMalloc((void**)&ds, buf_size);
    assign_ans<<<numBlocks, threads_per_block>>>(da, dc, ds, n);

    float last_element = 0;
    cudaMemcpy(&last_element, ds + n - 1, sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Send(&last_element, 1, MPI_FLOAT, 0, 102, MPI_COMM_WORLD);
    if (rank != 1) {
        float addition=0;
        MPI_Recv(&addition, 1, MPI_FLOAT, 0, 103, MPI_COMM_WORLD, &mpi_status);
        sequential_add<<<numBlocks, threads_per_block>>>(ds, n, addition);
    }
    
    cudaMemcpy(s, ds, buf_size, cudaMemcpyDeviceToHost);
    s[0] = a[0];
    MPI_Send(s, n, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);

    cudaFree(da);
    cudaFree(ds);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cout << cudaEventElapsedTime(&milliseconds, start, end);
}

float* parscan_cuda(float* a, int n){
    if (n==1) return a;
    int threads_per_block=8;
    int numBlocks = (n + threads_per_block - 1) / threads_per_block;
    
    int m = n/2;
    float* b = (float*)malloc(m*sizeof(float));
    float* da;
    cudaMalloc((void**)&da, n*sizeof(float));
    float* db;
    cudaMalloc((void**)&db, m*sizeof(float));
    cudaMemcpy(da, a, n*sizeof(float), cudaMemcpyHostToDevice);
    pair_sum<<<numBlocks, threads_per_block>>>(da, db, m);
    cudaMemcpy(b, db, m*sizeof(float), cudaMemcpyDeviceToHost);
    
    float* c = parscan_cuda(b, m);
    float* dc;
    cudaMalloc((void**)&dc, m * sizeof(float));
    cudaMemcpy(dc, c, m*sizeof(float), cudaMemcpyHostToDevice);
    float* ds;
    cudaMalloc((void**)&ds, n * sizeof(float));
    assign_ans<<<numBlocks, threads_per_block>>>(da, dc, ds, n);
    float* s = (float*)malloc(n * sizeof(float));
    cudaMemcpy(s, ds, n*sizeof(float), cudaMemcpyDeviceToHost);
    return s;
}


void seq_scan_mpicuda(int n, int rank) {
    
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    const int threads_per_block = 8;
    int numBlocks = (n + threads_per_block  - 1) / threads_per_block;
    
    size_t buf_size = n*sizeof(float);
    float* a = (float*)malloc(buf_size);
    float* s = (float*)malloc(buf_size);

    MPI_Status mpi_status;
    float* da;
    cudaMalloc((void**)&da, buf_size);
    MPI_Recv(a, n, MPI_FLOAT, 0, 100, MPI_COMM_WORLD, &mpi_status);
    // for (int i=0; i < n; i++){
    //     printf("cuda program input:%f in rank %d", a[i], rank);
    // }
    cudaMemcpy(da, a, buf_size, cudaMemcpyHostToDevice);
    float* ds;
    cudaMalloc((void**)&ds, buf_size);
    sequential_sum<<<numBlocks, threads_per_block>>>(da, ds, n);

    float last_element =0;
    cudaMemcpy(&last_element, ds + n - 1, sizeof(float), cudaMemcpyDeviceToHost);

    MPI_Send(&last_element, 1, MPI_FLOAT, 0, 102, MPI_COMM_WORLD);
    if (rank != 1) {
        float addition=0;
        MPI_Recv(&addition, 1, MPI_FLOAT, 0, 103, MPI_COMM_WORLD, &mpi_status);
        sequential_add<<<numBlocks, threads_per_block>>>(ds, n, addition);
    }
    
    cudaMemcpy(s, ds, buf_size, cudaMemcpyDeviceToHost);
    s[0] = a[0];
    // printf("before sending local output to master\n");
    MPI_Send(s, n, MPI_FLOAT, 0, 101, MPI_COMM_WORLD);
    // for (int i=0; i < n; i++){
    //     printf("cuda program output:%f in rank %d", s[i], rank);
    // }
    // printf("after sending local output to master\n");

    cudaFree(da);
    cudaFree(ds);
    
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float milliseconds = 0;
    cout << cudaEventElapsedTime(&milliseconds, start, end);    
}

float* seq_scan_cuda(float *a, int n) {
    
    if (n == 1) {
        return a;
    }
    
    const int threads_per_block = 8;
    int numBlocks = (n + threads_per_block  - 1) / threads_per_block;
    
    size_t buf_size = n*sizeof(float);
    float* s = (float*)malloc(buf_size);
    s[0] = a[0];

    float* da;
    cudaMalloc((void**)&da, buf_size);
    cudaMemcpy(da, a, buf_size, cudaMemcpyHostToDevice);
    float* ds;
    cudaMalloc((void**)&ds, buf_size);
    cudaMemcpy(ds, s, buf_size, cudaMemcpyHostToDevice);
    sequential_sum<<<numBlocks, threads_per_block>>>(da, ds, n);
    cudaMemcpy(s, ds, buf_size, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(ds);
    cudaDeviceReset();
    return s;
    
}

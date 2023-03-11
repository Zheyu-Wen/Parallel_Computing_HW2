#include <set>
#include <fstream>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

#include <omp.h>
#include "parscan.cuh"

float* sequential_scan3D(float* a, int n) {
   int total_dim = pow(n, 3);
   float* s = new float[total_dim];
   if (n == 1) {
      return a;
   }
   s[0] = a[0];
   // first: 1 dim prefix sum
   #pragma omp parallel for
   for (int i=1; i<n; i++){
      s[i*n*n] = s[(i-1)*n*n] + a[i*n*n]; 
   }
   
   #pragma omp parallel for
   for (int j=1; j<n; j++){
      s[j*n] = s[(j-1)*n] + a[j*n]; 
   }
   
   #pragma omp parallel for
   for (int k=1; k<n; k++){
      s[k] = s[k-1] + a[k]; 
   }

   // second: 2 dim prefix sum
   #pragma omp parallel for
   for (int i=1; i<n; i++){
      for (int j=1; j<n; j++){
         s[i*n*n + j*n] = s[(i-1)*n*n + j*n] + s[i*n*n + (j-1)*n] - s[(i-1)*n*n + (j-1)*n] + a[i*n*n + j*n]; 
      }  
   }
   
   #pragma omp parallel for
   for (int j=1; j<n; j++){
      for (int k=1; k<n; k++){
         s[j*n + k] = s[(j-1)*n + k] + s[j*n + k - 1] - s[(j-1)*n + k - 1] + a[j*n + k]; 
      }  
   }
   
   #pragma omp parallel for
   for (int k=1; k<n; k++){
      for (int i=1; i<n; i++){
         s[i*n*n + k] = s[(i-1)*n*n + k] + s[i*n*n + k - 1] - s[(i-1)*n*n + k - 1] + a[i*n*n + k]; 
      }  
   }

   // third: 3 dim prefix sum
   #pragma omp parallel for
   for (int i=1; i<n; i++){
      for (int j=1; j<n; j++){
         for (int k=1; k<n; k++){
            s[i*n*n + j*n + k] = s[(i-1)*n*n + j*n + k] + s[i*n*n + (j-1)*n + k] + s[i*n*n + j*n + k - 1] 
                                 - s[(i-1)*n*n + (j-1)*n + k] - s[(i-1)*n*n + j*n + k-1] - s[i*n*n + (j-1)*n + k-1]
                                 + s[(i-1)*n*n + (j-1)*n + k - 1] + a[i*n*n + j*n + k]; 
         }
      }  
   }
   return s;
}

float* parscan_cpu_openmp(float *a, int n) {
   float* s = nullptr;
   if (n == 1) {
      s = new float[1];
      s[0] = a[0];
      return s;
   }
   int m = n/2;
   float* b = new float[m];
   s = new float[n];

   #pragma omp parallel for
   for(int i=0;i<m;i++) 
   {
      b[i] = a[2*i] + a[2*i+1];
   }
   float* c = parscan_cpu_openmp(b, m);
   s[0] = a[0];

   #pragma omp parallel for
   for(int i=1;i<n;i++){
      if(i%2==1) s[i] = c[(i-1)/2];
      else s[i] = a[i] + c[i/2-1];
   }
//  delete[] b;
//  delete[] c;
   return s;
}

void print_results(const float* output, const size_t n, const size_t start_idx=0, const size_t end_idx=16) {
   for (int i = start_idx; i < end_idx; i++){
      cout << output[i] << " ";
   }
   cout << endl;
}

int main(int argc, char *argv[]){
   
   int N = atoi(argv[3]);	
   unsigned long long int n = pow(2, N);
   float* test_input;
   void* mapped_mem;
   const size_t file_size = n * sizeof(float);
   int fd;
   if (n >= pow(2, 35)){
      
      const char* file_path = "/scratch/08171/zheyw1/Parallel_Computing_HW/hw2/temp_storage_file";
      fd = open(file_path, O_RDWR | O_CREAT | O_TRUNC, 0666);
      lseek(fd, file_size-1, SEEK_SET);
      write(fd,"",1);

      mapped_mem = mmap(nullptr, file_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (mapped_mem == MAP_FAILED){
         std::cerr << "failed to map memory-mapped file: " << strerror(errno) << std::endl;
         return 1;
      }
      test_input = static_cast<float*>(mapped_mem);
   }
   else{
      test_input = (float*)malloc(n*sizeof(float));
      if (test_input == nullptr) {
         perror("malloc failed!");
         exit(1);
      }
   }
   
   srand(time(0));
   #pragma omp parallel for
   for (int i=0; i<n; ++i){
       test_input[i] = 1; //rand()%100;
       //if (i < 16) cout<< "main input:" << test_input[i]<<endl;
   }

   // if argv[1]=="cpu", then the code will run openmp
   // if argv[1]=="gpu", then the code will run mpi and cuda
   if (std::string(argv[2]).compare("3D") == 0){
      int total_dim = pow(n, 3);
      float* test_input_3D = new float[total_dim];
      srand(time(0));
      for (int i=0;i<n;i++) {
         for (int j=0;j<n;j++){
            for (int k=0;k<n;k++){
               test_input_3D[i*n*n + j*n + k] = 1;
            }
         }
      }
      auto start = tic;
      float* output = sequential_scan3D(test_input_3D, n);
      auto end = toc;
      auto time = milliseconds(end - start);
      cout<<time.count()<<"milliseconds"<<endl;
      print_results(output, n); 
      return 0;
   }

   if (std::string(argv[1]).compare("cpu") == 0){
      auto start = tic;
      float* output = parscan_cpu_openmp(test_input, n);
      auto end = toc;
      auto time = milliseconds(end - start);
      cout<<time.count()<<"milliseconds"<<endl;
      //print_results(output, n, 0, 16);
      return 0;
   }

   int rank, size;
   MPI_Init(&argc, &argv);
   MPI_Comm comm = MPI_COMM_WORLD;
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   
   std::set<int> workers = {};
   for (int i=1; i<size; i++) workers.emplace(i);

   auto start = tic;
   size_t n_chunck = size - 1;
   size_t chunck_size = n / n_chunck;
   // TODO fix not integer division case
   size_t local_buf_size = chunck_size * sizeof(float);
   size_t global_buf_size = n * sizeof(float);

   float* output = (float*)malloc(global_buf_size);
   float* array_last = (float*)malloc(n_chunck*sizeof(float));

   
   if (rank == 0) {
      // cout << endl;
      // printf("input_ptr = %p\n", test_input);
      for (const int &worker : workers) {
         float* start_input_ptr = test_input + (worker-1)*chunck_size;
         // printf("worker = %d, start_input_ptr = %p\n", worker, start_input_ptr);
         MPI_Send(start_input_ptr, chunck_size, MPI_FLOAT, worker, 100, MPI_COMM_WORLD);
      }

      for (const int &worker : workers) {
         float* array_last_element = array_last + (worker-1);
         // printf("worker = %d, start_output_ptr = %p\n", worker, start_output_ptr);
         MPI_Status mpi_status;
         MPI_Recv(array_last_element, 1, MPI_FLOAT, worker, 102, MPI_COMM_WORLD, &mpi_status);
      }

      float* scan_array = seq_scan_cuda(array_last, n_chunck);
      // for (int i=0; i<n_chunck; i++) printf("scan array last %f \n",scan_array[i]);

      for (auto it = workers.begin(); it != workers.end(); ++it) {
         if (it == workers.begin()) { continue; }
         int worker = *it;
         // printf("current worker: %d. \n", worker);
         float* start_input_ptr = scan_array + worker - 2;
         MPI_Send(start_input_ptr, 1, MPI_FLOAT, worker, 103, MPI_COMM_WORLD);
      }

      for (const int &worker : workers) {
         float* start_output_ptr = output + (worker-1)*chunck_size;
         // printf("worker = %d, start_output_ptr = %p\n", worker, start_output_ptr);
         MPI_Status mpi_status;
         MPI_Recv(start_output_ptr, chunck_size, MPI_FLOAT, worker, 101, MPI_COMM_WORLD, &mpi_status);
      }
   }

   // MPI_Barrier(comm);
   if (workers.find(rank) != workers.end()) {
      // cout << "before seq_scan_mpi: rank = " << rank << endl;
      // seq_scan_mpicuda(chunck_size, rank);
      parscan_mpicuda(chunck_size, rank);
      // cout << "after seq_scan_mpi: rank = " << rank << endl;
   }
   // MPI_Barrier(comm);


   auto end = toc;
   auto time = milliseconds(end - start);
   if (rank == 0) {
      cout<<time.count()<<"milliseconds"<<endl;
      print_results(output, n, 0, 16);
   }
   //cout << sizeof(output) << endl;
   delete[] output;
   cudaDeviceReset();
   MPI_Finalize();
   if (n >= pow(2, 35)){
      munmap(mapped_mem, file_size);
      close(fd);
   }
   return 0;
}


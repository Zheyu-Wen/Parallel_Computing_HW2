#!/bin/bash
#SBATCH -J CUDA_MPI
#SBATCH -o /home1/08171/zheyw1/Parallel_Computing_HW/hw2-sequential_CUDA_MPI_OpenMP_Scan1D3D/log_cudampi
#SBATCH -n 3
#SBATCH -N 3
#SBATCH -p gpu-a100
#SBATCH -t 00:10:00
#SBATCH -A TRA23001 

source ~/.bashrc
module load launcher
export OMP_NUM_THREADS=25

mpirun -n 3 bin/parscan_main gpu 1D 20


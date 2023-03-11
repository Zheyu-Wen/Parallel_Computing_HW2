#!/bin/bash
#SBATCH -J OpenMP
#SBATCH -o /home1/08171/zheyw1/Parallel_Computing_HW/hw2-sequential_CUDA_MPI_OpenMP_Scan1D3D/log_cpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p gpu-a100
#SBATCH -t 00:10:00
#SBATCH -A TRA23001 

source ~/.bashrc
module load launcher
export OMP_NUM_THREADS=25

bin/parscan_main cpu 1D 34


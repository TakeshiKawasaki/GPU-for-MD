#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -q gpu.q@langevin                                                                                                                                                          
#$ -l gpu=1
export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
nvidia-cuda-mps-control -d
cd /home/kawasaki/data/gpu/AQS/0.845_N9728mpi

mpirun -n 5 FITR_jamming_AQS_shear_thrust_mpi.out  # execute file

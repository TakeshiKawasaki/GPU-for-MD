#!/bin/sh

#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -q gpu.q
#$ -l gpu=1

export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
./a.out

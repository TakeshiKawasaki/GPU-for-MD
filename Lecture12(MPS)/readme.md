To confirm the version of your mpi, execute <br>
`mpi -showme`


A shell script for compilation:
```
nvcc  FIRE_shear_mpi.cu  -o  FIRE_shear_mpi.out -I/home/appl/openmpi-4.0.5-ic1912/include   -L/home/appl/openmpi-4.0.5-ic1912/lib -lmpi
```
A qsub code is:
```
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
```


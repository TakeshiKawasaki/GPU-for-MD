Typical results:<br>


<img width="800" alt="ed300abd-51f7-4bd7-b462-2ae72636ec8a" src="https://github.com/TakeshiKawasaki/GPU-for-MD/assets/72239760/23d7b527-24d6-4081-8ebb-ca4f358bad82">


To confirm the version of your mpi, execute <br>
`mpicc -showme`
A result would be:
```
icc -I/home/appl/openmpi-4.0.5-ic1912/include -pthread -Wl,-rpath -Wl,/home/appl/openmpi-4.0.5-ic1912/lib -Wl,--enable-new-dtags -L/home/appl/openmpi-4.0.5-ic1912/lib -lmpi
```

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
# cd /home/kawasaki/data/gpu/AQS/0.845_N9728mpi

mpirun -n 5 FITR_jamming_AQS_shear_thrust_mpi.out  # execute file
```


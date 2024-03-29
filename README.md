# GPU-for-MD <br>
<img width="668" alt="4818c976-50e6-4f5b-9ac0-9e43b5b1d55b" src="https://github.com/TakeshiKawasaki/GPU-for-MD/assets/72239760/37434823-460d-4784-b4ca-933c89af0b5b">


This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].<br>
[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

The first half of this repository is based on a tutorial (https://physpeach.github.io/cuda-tutorial/) by Shukawa-san, a lab graduate. 
The latter is my original， especially on the molecular dynamics codes. 

## Obtaining the codes via Git (available in our clusters) <br>
 `git clone https://github.com/TakeshiKawasaki/GPU-for-MD repo` <br>
 For update:<br>
 `git pull` 
 
## Compilation with Cuda <br>
 `nvcc add.cu -o add.out` 

## Execution <br>
`./add.out` 

## Job submission <br>
`qsub qsub.sh` 

```shell:qsub.sh
#!/bin/sh                                                                                                                 

#$ -S /bin/sh                                                                                                             
#$ -cwd                                                                                                                   
#$ -V                                                                                                                     
#$ -q gpu.q                                                                                                               
#$ -l gpu=1                                                                                                               

export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
./a.out # change the name
```

This is for mpi + GPU (by Jeong-san)
```shell:qsub_mpi.sh
#!/bin/sh
#$ -S /bin/sh
#$ -cwd
#$ -V
#$ -q gpu.q
#$ -l gpu=1
export CUDA_VISIBLE_DEVICES=$SGE_HGR_gpu
nvidia-cuda-mps-control -d
mpirun -n 2 FIRE_mpi.out # execute file
```

Compile for mps (by Jeong-san)
```
nvcc FIRE_mpi.cu -o FIRE_mpi.out 
-I/home/appl/openmpi-1.6.4-ic1217/include 
-L/home/appl/openmpi-1.6.4-ic1217/lib -I ~/inlcude -L ~/lib -lsprng -lcudart -lmpi
```



## Lecture 1 <br>
Basic operation of Cuda [1]
## Lecture 2 <br>
Reduction [1]
## Lecture 3 <br>
Generation of Gaussian random numbers [1]
## Lecture 4 <br>
Langevin dynamics without interactions
## Lecture 5 <br>
Langevin dynamics with interactions
## Lecture 6 <br>
Langevin dynamics with interactions adopting automatic updating Verlet list 
## Lecture 7 <br>
Langevin dynamics with interactions adopting automatic updating cell list

## Lecture 8 <br>
Molecular dynamics (MD) with interactions adopting automatic updating cell list

## Lecture 9 <be>
FIRE algorithm for getting jamming configurations.

## Lecture 10 <be>
FIRE algorithm for AQS shear.

## Lecture 11 <be>
FIRE algorithm for AQS shear + shear stabilization.

## Lecture 12 <be>
FIRE algorithm for AQS shear with MPI (multi ensembles).

## Lecture 13 <be>
FIRE algorithm for constant pressure AQS.

## Reference <br>
[1] https://physpeach.github.io/cuda-tutorial/ <br>
[2] NVIDIA's instructions:  https://http.download.nvidia.com/developer/cuda/jp/CUDA_Programming_Basics_PartI_jp.pdf <br>
[3] BOOK: "CUDA by example" 汎用GPUプログラミング入門 インプレスジャパン  Github:https://github.com/CodedK/CUDA-by-Example-source-code-for-the-book-s-examples-


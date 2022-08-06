# GPU-for-MD <br>
This work is licensed under a
[Creative Commons Attribution-ShareAlike 4.0 International License][cc-by-sa].<br>
[![CC BY-SA 4.0][cc-by-sa-image]][cc-by-sa]

[cc-by-sa]: http://creativecommons.org/licenses/by-sa/4.0/
[cc-by-sa-image]: https://licensebuttons.net/l/by-sa/4.0/88x31.png
[cc-by-sa-shield]: https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg

The first half of this repository is based on a tutorial (https://physpeach.github.io/cuda-tutorial/) by Shukawa-san, a graduate of our lab. 
The latter half， especially on the molecular dynamics code is my original. Automatic update of the cell list in GPU　is quite essential.

##　Obtaining the codes via Git (available in our clusters) <br>
 `git clone https://github.com/TakeshiKawasaki/GPU-for-MD` <br>
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
./add.out
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
Langevin dynamics with interactions adopting automatic updateing Verlet list 
## Lecture 7 <br>
Langevin dynamics with interactions adopting automatic updateing Cell list

## Reference <br>
[1] https://physpeach.github.io/cuda-tutorial/

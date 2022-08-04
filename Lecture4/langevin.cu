#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "../timer.cuh"
#include <math.h>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include "MT.h"
using namespace std;
//Using "const", the variable is shared into both gpu and cpu. 
const int  NT = 1024; //Num of the cuda threads.
const int  NP = 1e+6; //Particle number.
const int NB = (NP+NT-1)/NT; //Num of the cuda blocks.
const double dt= 0.01;
const int timemax = 1e+3;
//Langevin parameters
const double zeta = 1.0;
const double temp = 1.0;

//Initiallization of "curandState"
__global__ void setCurand(unsigned long long seed, curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(seed, i_global, 0, &state[i_global]);
}

//Gaussian random number's generation
__global__ void genrand_kernel(float *result, curandState *state){  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  result[i_global] = curand_normal(&state[i_global]);
}

//Gaussian random number's generation
__global__ void langevin_kernel(double*x_dev,double *v_dev,curandState *state, double noise_intensity){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    //  printf("%d,%f\n",i_global,v_dev[i_global]);
    v_dev[i_global] += -v_dev[i_global]*dt+ noise_intensity*curand_normal(&state[i_global]);
    x_dev[i_global] += v_dev[i_global]*dt;
  }
}


void init_array(double *x,int Narr, double c){
  for(int i=0;i<Narr;i++) x[i] = c;
}

int main(){
  double *x,*v,*x_dev,*v_dev;
  curandState *state; //Cuda state for random numbers
  double sec; //measurred time
  double noise_intensity = sqrt(2.*zeta*temp*dt); //Langevin noise intensity.  
  x = (double*)malloc(NB*NT*sizeof(double));
  v = (double*)malloc(NB*NT*sizeof(double));
  cudaMalloc((void**)&x_dev, NB * NT * sizeof(double)); // CudaMalloc should be executed once in the host. 
  cudaMalloc((void**)&v_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&state, NB * NT * sizeof(curandState)); 
  init_array(x,NT*NB,0.);
  init_array(v,NT*NB,0.);
  cudaMemcpy(x_dev, x, NB * NT* sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(v_dev, v, NB * NT* sizeof(double),cudaMemcpyHostToDevice);  
  setCurand<<<NB,NT>>>(0, state);
 
  for(double t=0;t<timemax;t+=dt){     
    langevin_kernel<<<NB,NT>>>(x_dev,v_dev,state,noise_intensity);
    //  cudaDeviceSynchronize(); // for printf in the device.
  } 
  cudaMemcpy(x, x_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(v, v_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  //  double v_s= sum(v);
  double diffusion=0.0;
  for(int i=0;i<NP;i++)
    diffusion += 0.5*x[i]*x[i]/timemax/NP;
    
  cout <<diffusion<<endl;

  //  cout << sec <<"sec"<<endl;  
  cudaFree(x_dev);
  cudaFree(v_dev);
  cudaFree(state);
  free(x); 
  free(v); 
  return 0;
}

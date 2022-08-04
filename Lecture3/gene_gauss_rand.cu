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
//num of thread
#define NT 1024
//num of block
//#define NB 100
#define N  1e+6
int NB = (int)((N+NT-1)/NT);

double unif_rand(double left, double right)
{
  return left + (right - left) * rand() / RAND_MAX;
}
double gauss_rand_cpu(void)
{
  static double iset = 0;
  static double gset;
  double fac, rsq, v1, v2;
  if (iset == 0) {
    do {
      v1 = unif_rand(-1, 1);
      v2 = unif_rand(-1, 1);
      rsq = v1 * v1 + v2 * v2;
    } while (rsq >= 1.0 || rsq == 0.0);
    fac = sqrt(-2.0 * log(rsq) / rsq);
    gset = v1 * fac;
    iset = 0.50;
    return v2 * fac;
  }
  else {
    iset = 0;
    return gset;
  }
}

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

// Generator function for gaussian rand vector rnd[NT*NB];

void gaussrand(float *rnd){
  float *rnd_dev;
  curandState *state;
  double sec;
   
  cudaMalloc((void**)&rnd_dev, NB * NT * sizeof(double)); // spending 0.2sec
  cudaMalloc((void**)&state, NB * NT * sizeof(curandState)); //spending 0.2sec

  setCurand<<<NB,NT>>>(0, state); // "0" is a seed.
  measureTime(); 
  for(int i=0;i<1000;i++)     
    genrand_kernel<<<NB,NT>>>(rnd_dev,state); 
  sec = measureTime()/1000;

  cudaMemcpy(rnd, rnd_dev, NB * NT* sizeof(uint),cudaMemcpyDeviceToHost);
  
  cout << sec <<"sec"<<endl;  
  cudaFree(rnd_dev);
  cudaFree(state);
}


int main(){
  float *rnd;
  rnd = (float*)malloc(NB*NT*sizeof(float));
  
  double av = 0., sigma = 0.,sec;
  
  gaussrand(rnd);
  
  for(uint i = 0; i < N; i++)
    av += rnd[i]/N;
  for(uint i = 0; i < N; i++)
    sigma += (rnd[i]-av)*(rnd[i]-av)/N;  
  sigma = sqrt(sigma);
  
  printf("\nav = %f\nsig = %f\n",av,sigma);
  
  measureTime();
  for(uint i = 0; i < N; i++)
    gauss_rand_cpu();
  
  
  sec = measureTime()/1000;
  cout << sec <<"sec"<<endl;  
  
  free(rnd);  
  return 0;
}

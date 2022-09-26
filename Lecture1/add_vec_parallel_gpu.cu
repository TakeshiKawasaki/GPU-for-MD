#include <iostream>
#include <cuda.h>
#include "timer.cuh"

using namespace std;

const int Nv = 1000000;
const int NT = 512; //The number of threads
const int NB = (Nv + NT - 1)/NT;

void setup_vec(int *vec, int a){
  for(int i = 0; i < Nv; i++){

    vec[i] = i * a;
  }
  return;
}


__global__ void add_vec(int *c, int *a, int* b){
  //  int i_global = threadIdx.x; 
  int i_global = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_global < Nv)
    c[i_global] = a[i_global] + b[i_global];
}

int main(){
  int *a, *b, *c;
  int *a_dev, *b_dev, *c_dev;

  //Allocation
  a = (int*)malloc(Nv * sizeof(int));
  b = (int*)malloc(Nv * sizeof(int));
  c = (int*)malloc(Nv * sizeof(int));


  cudaMalloc((void**)&a_dev, Nv * sizeof(int));
  cudaMalloc((void**)&b_dev, Nv * sizeof(int));
  cudaMalloc((void**)&c_dev, Nv * sizeof(int));

  //Setup input vecs
  setup_vec(a, 1);
  setup_vec(b, 2);

  //Transfer input vecs to device
  cudaMemcpy(a_dev, a, Nv * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, b, Nv * sizeof(int), cudaMemcpyHostToDevice);

  //Launch add_vec() on the device
  double ms;
  measureTime();
  for(int i = 0; i < 100000; i++){
    add_vec<<<NB, NT>>>(c_dev, a_dev, b_dev);
   }

  cudaMemcpy(c, c_dev, Nv *sizeof(int), cudaMemcpyDeviceToHost);
  ms = measureTime();


  cout << "Time: " << ms/1000. << "ms" << endl;

  cout << "c: " << c[10000] <<  endl;

  //Free
  free(a);
  free(b);
  free(c);
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);

  return 0;
}

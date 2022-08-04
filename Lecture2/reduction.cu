#include <iostream>
#include <cuda.h>
#include "timer.cuh"

using namespace std;

const int Narr = 1.e+8;
const int NT = 1024; //The number of threads
//const int NB = (Nv + NT - 1)/NT;

void setup_vec(double *vec, int a){
  for(int i = 0; i < Narr; i++){
    vec[i] = (double)i * 1.0;
  }
  return;
}

__global__ void addReduction(double *out, double *in, int remain, int reduce){
  int i_global = blockIdx.x * blockDim.x + threadIdx.x;

  //  printf("Hello World from block %d, thread %d\n", blockIdx.x, threadIdx.x);
  out[i_global] = in[i_global];
  if(i_global < reduce){
    out[i_global] += in[i_global + remain];
  }

}


double sum(double *x){
  double  sum;
  //Allocate 2 arrs for reduction
  double *x_dev[2];
  cudaMalloc((void**)&x_dev[0], Narr * sizeof(double));
  cudaMalloc((void**)&x_dev[1], Narr * sizeof(double));
  cudaMemcpy(x_dev[0], x, Narr * sizeof(double), cudaMemcpyHostToDevice);
  int remain, reduce;
  int flip = 0;

  //Reduction operation
  for(int len = Narr; len > 1; len -= reduce){
    reduce = len >> 1; //reduce = len / 2
    remain = len - reduce;
    flip = !flip; // 1 == !(0), 0 == !(1)
    addReduction<<<(remain + NT - 1)/NT,NT>>>(x_dev[flip], x_dev[!flip], remain, reduce);
  }
  cudaMemcpy(&sum, x_dev[flip], sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(&x_dev[0]);
  cudaFree(&x_dev[1]);
  return sum;
}


int main(){
 double *a;
 double sum_host;
 int i;
  
  //Allocation
  a = (double*)malloc(Narr * sizeof(double));
  
  //Setup input vecs
  setup_vec(a, 1);
  
  double ms;

  measureTime();

  for(i=0;i<1000;i++)
    sum_host=sum(a);
  
  ms = measureTime();
  
  cout << "Time: " << ms/1000. << "sec" << endl;
  cout << std::fixed << "sum:" << sum_host <<  endl;
  cout << std::scientific << "sum:" << sum_host <<  endl;
  
  //Free
  free(a);
  return 0;
}

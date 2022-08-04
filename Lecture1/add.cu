#include <iostream>
#include <cuda.h>
using namespace std;

__global__ void add(int *c, int *a, int *b){
  *c = *a + *b;
}

int main(){
  int a, b, c; //values on the host                                             
  int *a_dev, *b_dev, *c_dev; //values on the device                            

  //Allocate memories on the devices                                            
  cudaMalloc((void**)&a_dev, sizeof(int));
  cudaMalloc((void**)&b_dev, sizeof(int));
  cudaMalloc((void**)&c_dev, sizeof(int));

  //Setup input values                                                          
  a = 2;
  b = 7;

  //Transfer the inputs to device                                               
  cudaMemcpy(a_dev, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(b_dev, &b, sizeof(int), cudaMemcpyHostToDevice);

  //Launch add() kernel on the device                                           
  add<<<1, 1>>>(c_dev, a_dev, b_dev);

  //Transfer the output to host                                                 
  cudaMemcpy(&c, c_dev, sizeof(int), cudaMemcpyDeviceToHost);
  cout << "c: " << c << endl;

  //Free                                                                        
  cudaFree(a_dev);
  cudaFree(b_dev);
  cudaFree(c_dev);
}

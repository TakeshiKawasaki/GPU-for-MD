#include <iostream>
using namespace std;
__global__ void myKernel(){
  //nothing here!!
}

int main(){
  cout<<"hello CUDA"<< endl;
  myKernel<<<1,1>>>endl;
  return 0;
}

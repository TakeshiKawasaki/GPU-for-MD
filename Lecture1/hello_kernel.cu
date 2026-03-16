#include <iostream>
using namespace std;
__global__ void myKernel(){
  //nothing here!!
}

int main(){
  cout<<"hello CUDA"<< endl;
  myKernel<<<1,1>>>();
  return 0;
}

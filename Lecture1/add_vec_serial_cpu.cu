#include <iostream>
#include <cuda.h>
#include "timer.cuh"

using namespace std;

const int Nv = 1000000;

void setup_vec(int *vec, int a){
  for(int i = 0; i < Nv; i++){
    vec[i] = i * a;
  }
  return;
}

void add_vec(int *c, int *a, int* b){
  for(int i = 0; i < Nv; i++){
    c[i] = a[i] + b[i];
  }
  return;
}

int main(){
  int *a, *b, *c;

  //Allocation
  a = (int*)malloc(Nv * sizeof(int));
  b = (int*)malloc(Nv * sizeof(int));
  c = (int*)malloc(Nv * sizeof(int));

  //Setup input vecs
  setup_vec(a, 1);
  setup_vec(b, 2);

  //Launch add_vec()
  double ms;
  measureTime();
  for(int i = 0; i < 1000; i++){
    add_vec(c, a, b);
  }
  ms = measureTime();
  cout << "Time: " << ms/1000. << "ms" << endl;

  //Free
  free(a);
  free(b);
  free(c);

  return 0;
}

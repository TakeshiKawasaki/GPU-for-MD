#include <cuda.h>
#include <stdio.h>
#include <time.h>
#include "../timer.cuh"
#include <math.h>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>
#include "../MT.h"
using namespace std;

//Using "const", the variable is shared into both gpu and cpu.
const int  NT = 1024; //Num of the cuda threads.
const int  NP = 1e+4; //Particle number.
const int  NB = (NP+NT-1)/NT; //Num of the cuda blocks.
const double dt = 0.01;
const int timemax = 1.e+2;
//Langevin parameters
const double zeta = 1.0;
const double temp = 1.e-4;
const double rho = 0.85;
//const double LB = 100.;


//Initiallization of "curandState"
__global__ void setCurand(unsigned long long seed, curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(seed, i_global, 0, &state[i_global]);
}


//Gaussian random number's generation
__global__ void langevin_kernel(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,curandState *state, double noise_intensity,double LB){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    //  printf("%d,%f\n",i_global,v_dev[i_global]);
    vx_dev[i_global] += -zeta*vx_dev[i_global]*dt+ fx_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    vy_dev[i_global] += -zeta*vy_dev[i_global]*dt+ fy_dev[i_global]*dt + noise_intensity*curand_normal(&state[i_global]);
    x_dev[i_global] += vx_dev[i_global]*dt;
    y_dev[i_global] += vy_dev[i_global]*dt;

    x_dev[i_global]  -= LB*floor(x_dev[i_global]/LB);
    y_dev[i_global]  -= LB*floor(y_dev[i_global]/LB);
  //  printf("vx=%f\n",vx_dev[i_global]);
  }

}

//Force calculation NP*NP matrix...
__global__ void calc_force_kernel(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double LB){
  double dx,dy,dr,dU,a_ij;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;



  if(i_global<NP){
  fx_dev[i_global]=0.0;
  fy_dev[i_global]=0.0;
  for(int j = 0; j<NP; j++)
      if(j != i_global){
	dx=x_dev[j]-x_dev[i_global];
	dy=y_dev[j]-y_dev[i_global];

	dx -= LB*floor(dx/LB+0.5);
	dy -= LB*floor(dy/LB+0.5);

	dr = sqrt(dx*dx+dy*dy);
	a_ij=0.5*(a_dev[i_global]+a_dev[j]);

	if(dr < a_ij){
	  dU = -(1-dr/a_ij)/a_ij; //derivertive of U wrt r.
	  fx_dev[i_global] += dU*dx/dr;
	  fy_dev[i_global] += dU*dy/dr;
	}
      }
  // printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
}

__global__ void init_array_rand(double *x_dev, double c,curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c*curand_uniform(&state[i_global]);
}

void init_array(double *x,int Narr, double c){
  for(int i=0;i<Narr;i++)
    x[i] = c;
}

//void init_array_rand(double *x,int Narr, double c){
 // for(int i=0;i<Narr;i++)
   // x[i] = c*rand()/RAND_MAX;
//}

void output(double *x,double *y,double *vx,double *vy,double *a){
  static int count=1;
  char filename[128];
  sprintf(filename,"coord_%.d.dat",count);
  ofstream file;
  file.open(filename);
  double kine=0.0;

  for(int i=0;i<NP;i++){
    file << x[i] << " " << y[i]<< " " << a[i] << endl;
  //  cout << vx[i] << endl;

    kine+=0.5*(vx[i]*vx[i]+vy[i]*vy[i]);
  }
  file.close();
  cout<<"temp="<< kine/(double)NP <<endl;
  count++;
}


int main(){
  double *x,*vx,*y,*vy,*a,*x_dev,*vx_dev,*y_dev,*vy_dev,*a_dev,*fx_dev,*fy_dev;
  curandState *state; //Cuda state for random numbers
  double sec; //measurred time
  double noise_intensity = sqrt(2.*zeta*temp*dt); //Langevin noise intensity.
  double LB = sqrt(M_PI*1.0*1.0*(double)NP*0.25/rho);
  x  = (double*)malloc(NB*NT*sizeof(double));
  y  = (double*)malloc(NB*NT*sizeof(double));
  vx = (double*)malloc(NB*NT*sizeof(double));
  vy = (double*)malloc(NB*NT*sizeof(double));
  a  = (double*)malloc(NB*NT*sizeof(double));
  cudaMalloc((void**)&x_dev,  NB * NT * sizeof(double)); // CudaMalloc should be executed once in the host.
  cudaMalloc((void**)&y_dev,  NB * NT * sizeof(double));
  cudaMalloc((void**)&vx_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&vy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&fx_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&fy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&a_dev,  NB * NT * sizeof(double));
  cudaMalloc((void**)&state,  NB * NT * sizeof(curandState));

 // init_array_rand(x,NT*NB,LB);
 // init_array_rand(y,NT*NB,LB);
  init_array(a,NT*NB,1.0);
  init_array(vx,NT*NB,0.);
  init_array(vy,NT*NB,0.);

  cudaMemcpy(x_dev,   x, NB * NT* sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(y_dev,   y, NB * NT* sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vx_dev, vx, NB * NT* sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(vy_dev, vy, NB * NT* sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(a_dev,  a, NB * NT* sizeof(double),cudaMemcpyHostToDevice);

  setCurand<<<NB,NT>>>(0, state); // Construction of the cudarand state.
  init_array_rand<<<NB,NT>>>(x_dev,LB,state);
  init_array_rand<<<NB,NT>>>(y_dev,LB,state);

  for(double t=0;t<timemax;t+=dt){
    calc_force_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,LB);
    langevin_kernel<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,state,noise_intensity,LB);
    //     cudaDeviceSynchronize(); // for printf in the device.
  }
  cudaMemcpy(x,   x_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(vx, vx_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(y,   y_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(vy, vy_dev, NB * NT* sizeof(double),cudaMemcpyDeviceToHost);

  // for(int i=0;i<NP;i++)
  //  cout<<i<<" "<<vx[i]<<endl;

  output(x,y,vx,vy,a);

  cudaFree(x_dev);
  cudaFree(vx_dev);
  cudaFree(y_dev);
  cudaFree(vy_dev);
  cudaFree(state);
  free(x);
  free(vx);
  free(y);
  free(vy);
  return 0;
}

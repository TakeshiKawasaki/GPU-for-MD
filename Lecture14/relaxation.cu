#include <cuda.h>
#include <float.h>
#include <stdio.h>
#include <time.h>
#include "../timer.cuh"
#include <math.h>
#include <iostream>
#include <fstream>
#include <curand.h>
#include <curand_kernel.h>
#include "../MT.h"

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iomanip>
//#include <mpi.h>

using namespace std;

//Using "const", the variable is shared into both gpu and cpu. 
const int  NT = 1024; //Num of the cuda threads.
const int  NP = 50000; //Particle number.
const int  NB = (NP+NT-1)/NT; //Num of the cuda blocks.
const int  NN = 50;
const int  NPC = 100; // Number of the particles in the neighbour cell 
const double dt = 0.2;
const double RCHK =1.7;
const double rcut = 1.4;
const double phiini = 0.843;
const double f_thresh = 1.e-8;
const double p0 = 1.e-5;

//Initialization of "curandState"
__global__ void setCurand(unsigned long long seed, curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(seed, i_global, 0, &state[i_global]);
}


__global__ void disp_gate_kernel(double *vx_dev,double *vy_dev,double *dx_dev,double *dy_dev,int *gate_dev)
{
  double r2;  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(i_global<NP){
    dx_dev[i_global]+=vx_dev[i_global]*dt;
    dy_dev[i_global]+=vy_dev[i_global]*dt;
    r2 = dx_dev[i_global]*dx_dev[i_global]+dy_dev[i_global]*dy_dev[i_global];
    if(r2> 0.1*(RCHK-rcut)*(RCHK-rcut)){
      gate_dev[0]=1;
    }
  }
  __syncthreads();
}


__device__ int f(int i,int M){
  int k;
  k=i;
  if(k>=M)
    k-=M;
  if(k<0)
    k+=M;
  return k;
}


__global__ void cell_map(double *L_dev,double *x_dev,double *y_dev,int *map_dev,int *gate_dev, int M,double *gamma_dev)
{  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  int nx,ny;
  int num;
  if(gate_dev[0] == 1 && i_global<NP){
    nx = f((int)((x_dev[i_global]-(*gamma_dev)*y_dev[i_global])*(double)M/(double)(*L_dev)),M);
    ny = f((int)(y_dev[i_global]*(double)M/(double)(*L_dev)),M);
    num = atomicAdd(&map_dev[(nx+M*ny)*NPC],1);
    map_dev[(nx+M*ny)*NPC+num+1] = i_global;
  }
  __syncthreads();
}


__global__ void cell_list(double *L_dev,double *x_dev,double *y_dev,double *dx_dev,double *dy_dev,int *list_dev,int *map_dev,int *gate_dev, int M,double *gamma_dev)
{
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  int nx,ny;
  int j,k;
  double dx,dy,dy_temp,r2;  
  int l,m;
  if(gate_dev[0] == 1 && i_global<NP){
    list_dev[NN*i_global]=0;
    nx=f((int)((x_dev[i_global]-(*gamma_dev)*y_dev[i_global])*(double)M/(double)(*L_dev)),M);
    ny=f((int)(y_dev[i_global]*(double)M/(double)(*L_dev)),M);
    for(m=ny-1;m<=ny+1;m++)
      for(l=nx-1;l<=nx+1;l++)
	for(k=1; k<=map_dev[(f(l,M)+M*f(m,M))*NPC]; k++){
	  j = map_dev[(f(l,M)+M*f(m,M))*NPC+k];
	  if(j != i_global){
	    dx =x_dev[i_global] - x_dev[j];
	    dy =y_dev[i_global] - y_dev[j];

	    dy_temp=dy;
	    dy -=(*L_dev)*floor(dy/(*L_dev)+0.5);
	    dx -= *gamma_dev*(*L_dev)*floor((dy_temp+0.5*(*L_dev))/(*L_dev));
	    dx -=(*L_dev)*floor(dx/(*L_dev)+0.5);

	    r2 = dx*dx + dy*dy;
	    if(r2 < RCHK*RCHK){
	      list_dev[NN*i_global]++;
	      list_dev[NN*i_global+list_dev[NN*i_global]]=j;
	      // printf("i=%d, list=%d\n",i_global,list_dev[NN*i_global]);     
	    }
	  }
	}
    //    if(i_global == 0)
    //  printf("i=%d, list=%d\n",i_global,list_dev[NN*i_global]);      
    dx_dev[i_global]=0.;
    dy_dev[i_global]=0.;
  }
  __syncthreads();
  if(i_global == 0)
    gate_dev[0] = 0;
}


__global__ void calc_force_kernel(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double *L_dev,int *list_dev,double *f_dev, double *pressure_dev,double *sigma_dev){
  double dx,dy,dr,dU_r,a_ij;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    fx_dev[i_global]=0.0;
    fy_dev[i_global]=0.0;
    f_dev[i_global]=0.0;
    pressure_dev[i_global]=0.0;
    sigma_dev[i_global]=0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){
      dx = x_dev[list_dev[NN*i_global+j]]-x_dev[i_global];
      dy = y_dev[list_dev[NN*i_global+j]]-y_dev[i_global];
      dx -= (*L_dev)*floor(dx/(*L_dev)+0.5);
      dy -= (*L_dev)*floor(dy/(*L_dev)+0.5);	
      dr = sqrt(dx*dx+dy*dy);
      a_ij = 0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);
      if(dr < a_ij){
	dU_r = -(1-dr/a_ij)/a_ij; //derivertive of U wrt r.
	fx_dev[i_global] += dU_r*dx/dr;
	fy_dev[i_global] += dU_r*dy/dr;
	pressure_dev[i_global] -= dU_r*dr/(2.0**L_dev**L_dev);
	sigma_dev[i_global] += dU_r*dx*dy/dr/(*L_dev**L_dev);
      }
    }
    f_dev[i_global]+= sqrt(fx_dev[i_global]*fx_dev[i_global]+fy_dev[i_global]*fy_dev[i_global]);
    // printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
  __syncthreads();
}

__global__ void calc_energy_kernel(double*x_dev,double*y_dev,double *pot_dev,double *a_dev,double *L_dev,int *list_dev){
  double dx,dy,dr,a_ij;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    pot_dev[i_global]=0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){
      dx = x_dev[list_dev[NN*i_global+j]]-x_dev[i_global];
      dy = y_dev[list_dev[NN*i_global+j]]-y_dev[i_global];
      dx -= (*L_dev)*floor(dx/(*L_dev)+0.5);
      dy -= (*L_dev)*floor(dy/(*L_dev)+0.5);
      dr = sqrt(dx*dx+dy*dy);
      a_ij= 0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);
      if(dr < a_ij)
	pot_dev[i_global]+= 0.5*(1.-dr/a_ij)*(1.-dr/a_ij);
    }
  }
  __syncthreads();
}

__global__ void copy_kernel(double *x0_dev, double *y0_dev, double *x_dev, double *y_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x0_dev[i_global]=x_dev[i_global];
  y0_dev[i_global]=y_dev[i_global];
}

__global__ void init_gate_kernel(int *gate_dev, int c){
  gate_dev[0]=c;
}
__global__ void init_scalar_kernel(double *a_dev, double d){
  a_dev[0]=d;
}

__global__ void init_map_kernel(int *map_dev,int M){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  map_dev[i_global] = 0;
  __syncthreads();
}

__global__ void init_array(double *x_dev, double c){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c;
}

__global__ void init_diamters(double *a_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP/2)
    a_dev[i_global] = 1.0;
  if(i_global>NP/2 && i_global<NP)
    a_dev[i_global] = 1.4;
  __syncthreads();
}

__global__ void init_array_rand(double *x_dev, double c,curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c*curand_uniform(&state[i_global]);
}


__global__ void add_reduction(double *pot_dev, int *reduce_dev, int *remain_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global< *reduce_dev)
    pot_dev[i_global] += pot_dev[i_global+*remain_dev];

}

__global__ void len_ini(int *reduce_dev,int *remain_dev, int size){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global==0){
    *reduce_dev= size/2;
    *remain_dev= size - *reduce_dev; 
  }

}
__global__ void len_div(int *reduce_dev,int *remain_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global==0){
    *reduce_dev = *remain_dev/2;
    *remain_dev -= *reduce_dev;
  }
}

void list_auto_update(double *L_dev,double *x_dev, double *y_dev,double *vx_dev, double *vy_dev, double *dx_dev,double *dy_dev, int M, int *map_dev, int *gate_dev,int *list_dev, double *gamma_dev){
  disp_gate_kernel<<<NB,NT>>>(vx_dev,vy_dev,dx_dev,dy_dev,gate_dev);
  init_map_kernel<<<M*M,NPC>>>(map_dev,M);
  cell_map<<<NB,NT>>>(L_dev,x_dev,y_dev,map_dev,gate_dev,M,gamma_dev);
  cell_list<<<NB,NT>>>(L_dev,x_dev,y_dev,dx_dev,dy_dev,list_dev,map_dev,gate_dev,M,gamma_dev);
}

void add_reduction_array(double *array_dev, int *reduce_dev,int *remain_dev){
  len_ini<<<1,1>>>(reduce_dev,remain_dev,NP);
  int reduce=NP/2,remain=NP-NP/2;
  while(reduce>0){
    add_reduction<<<(reduce+NT-1)/NT,NT>>>(array_dev,reduce_dev,remain_dev);
    reduce = remain/2;remain-=reduce;
    len_div<<<1,1>>>(reduce_dev,remain_dev);
  }
}


__global__ void SD(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,double *L_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;

  if(i_global<NP){
    vx_dev[i_global] = fx_dev[i_global]*dt;
    vy_dev[i_global] = fy_dev[i_global]*dt;
    x_dev[i_global] += vx_dev[i_global]*dt;
    y_dev[i_global] += vy_dev[i_global]*dt;

    x_dev[i_global]  -= *L_dev*floor(x_dev[i_global]/(*L_dev));
    y_dev[i_global]  -= *L_dev*floor(y_dev[i_global]/(*L_dev));
  }
}


__global__ void SD_P(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,double *L_dev, double *pressure_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double vol_dot_dev,L_dev_temp;

  if(i_global<NP){
    L_dev_temp=*L_dev;
    vx_dev[i_global] = fx_dev[i_global]*dt;
    vy_dev[i_global] = fy_dev[i_global]*dt;
    x_dev[i_global] += vx_dev[i_global]*dt;
    y_dev[i_global] += vy_dev[i_global]*dt;

    vol_dot_dev = (double)NP*(pressure_dev[0] - p0)*dt;
    *L_dev += 0.5/L_dev_temp*(vol_dot_dev)*dt;

    x_dev[i_global] *= (*L_dev)/L_dev_temp;
    y_dev[i_global] *= (*L_dev)/L_dev_temp;

    x_dev[i_global]  -= *L_dev*floor(x_dev[i_global]/(*L_dev));
    y_dev[i_global]  -= *L_dev*floor(y_dev[i_global]/(*L_dev));

  }
}





int main(){
  double *x,*vx,*y,*vy,*pot,*x_dev,*vx_dev,*y_dev,*dx_dev,*dy_dev,*vy_dev,*pot_dev,*a_dev,*fx_dev,*fy_dev,*L_dev,*deltaphi_dev,deltaphi;
  double *f_dev,*pressure_dev,*sigma_dev,*phi_dev,*gamma_dev;
  int *list_dev,*map_dev,*gate_dev,*remain_dev,*reduce_dev;
  int *FIRE_gate_dev,FIRE_gate,*FIRE_param_gate_dev;
  int clock=0;
  curandState *state; //Cuda state for random numbers
  double sec; //measurred time
  double L = sqrt(M_PI*(1.0*1.0+1.4*1.4)*(double)NP/(8.*phiini));
  int M = (int)(L/RCHK);
  char filename[256];
  cout <<"M="<< M <<"L="<<L<<endl;
  
  x  = (double*)malloc(NB*NT*sizeof(double));
  y  = (double*)malloc(NB*NT*sizeof(double));
  vx = (double*)malloc(NB*NT*sizeof(double));
  vy = (double*)malloc(NB*NT*sizeof(double));
  pot  = (double*)malloc(NB*NT*sizeof(double));
  cudaMalloc((void**)&x_dev,  NB * NT * sizeof(double)); // CudaMalloc should be executed once
  cudaMalloc((void**)&y_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&dx_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&dy_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&fx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&fy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&f_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&pressure_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&sigma_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&pot_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&a_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&L_dev, sizeof(double)); 
  cudaMalloc((void**)&gamma_dev, sizeof(double)); 
  cudaMalloc((void**)&phi_dev, sizeof(double));
  cudaMalloc((void**)&deltaphi_dev, sizeof(double)); 
  cudaMalloc((void**)&gate_dev, sizeof(int));
  cudaMalloc((void**)&FIRE_gate_dev, sizeof(int)); 
  cudaMalloc((void**)&FIRE_param_gate_dev, sizeof(int)); 
  cudaMalloc((void**)&remain_dev, sizeof(int));
  cudaMalloc((void**)&reduce_dev, sizeof(int));
  cudaMalloc((void**)&list_dev,  NB * NT * NN* sizeof(int)); 
  cudaMalloc((void**)&map_dev,  M * M * NPC* sizeof(int)); 
  cudaMalloc((void**)&state,  NB * NT * sizeof(curandState)); 
  cudaMemcpy(L_dev, &L,sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(phi_dev,&phiini,sizeof(double),cudaMemcpyHostToDevice);
  setCurand<<<NB,NT>>>(0, state); // Construction of the cudarand state.  
  init_array_rand<<<NB,NT>>>(x_dev,L,state);
  init_array_rand<<<NB,NT>>>(y_dev,L,state);
  init_diamters<<<NB,NT>>>(a_dev);
  init_array<<<NB,NT>>>(vx_dev,0.);
  init_array<<<NB,NT>>>(vy_dev,0.);
  init_array<<<NB,NT>>>(pot_dev,0.);
  init_gate_kernel<<<1,1>>>(gate_dev,1);
  init_map_kernel<<<M*M,NPC>>>(map_dev,M);

  cell_map<<<NB,NT>>>(L_dev,x_dev,y_dev,map_dev,gate_dev,M,gamma_dev);
  cell_list<<<NB,NT>>>(L_dev,x_dev,y_dev,dx_dev,dy_dev,list_dev,map_dev,gate_dev,M,gamma_dev);  

  init_scalar_kernel<<<1,1>>>(deltaphi_dev,1.e-4);
  init_scalar_kernel<<<1,1>>>(gamma_dev,0.0);


  sprintf(filename,"phi_p%fN%d.dat",p0,NP);
  ofstream file;
  file.open(filename);
  file<<std::setprecision(12)<< "# force "<<" pressure " << " phi "<<endl;

  measureTime();  
  for(;;){
    clock++;
    calc_force_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,L_dev,list_dev,f_dev,pressure_dev,sigma_dev);
    double sum_f = thrust::reduce(thrust::device_pointer_cast(f_dev), thrust::device_pointer_cast(f_dev + NB * NT),0.0,thrust::plus<double>());
    double sum_pressure = thrust::reduce(thrust::device_pointer_cast(pressure_dev), thrust::device_pointer_cast(pressure_dev + NB * NT),0.0,thrust::plus<double>());
    cudaMemcpy(&pressure_dev[0],&sum_pressure, sizeof(double),cudaMemcpyHostToDevice); 

    SD<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,L_dev);
    //  SD_P<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,L_dev,pressure_dev);
    list_auto_update(L_dev,x_dev,y_dev,vx_dev,vy_dev,dx_dev,dy_dev,M,map_dev,gate_dev,list_dev,gamma_dev);

    // cudaMemcpy(&f_dev[0], &sum_f, sizeof(double), cudaMemcpyHostToDevice);
    if(clock%1000000 == 0){
      cudaMemcpy(&L,L_dev, sizeof(double),cudaMemcpyDeviceToHost);
      double phi = M_PI*(1.0*1.0+1.4*1.4)*(double)NP/(8.*L*L);
      cout<<std::setprecision(12)<< "count= "<< clock <<" f= "<<sum_f/NP <<" pressure= "<<sum_pressure <<" phi = "<<  phi <<endl;
      file<<std::setprecision(12)<< clock*dt <<" "<< sum_f/NP <<" "<<sum_pressure <<" "<<  phi <<endl;

      sec = measureTime()/1000;  
      cout<<"time(sec):"<< sec <<endl;
      measureTime();
    }
    if(sum_f/NP < f_thresh){
      // calc_energy_kernel<<<NB,NT>>>(x_dev,y_dev,pot_dev,a_dev,L_dev,list_dev);
      // add_reduction_array(pot_dev,reduce_dev,remain_dev);
      // cudaMemcpy(pot,pot_dev, NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
      // cudaMemcpy(&phi,phi_dev, sizeof(double),cudaMemcpyDeviceToHost);
      cout<<std::setprecision(12)<< "count= "<< clock <<" f= "<<sum_f/NP <<" pressure="<<sum_pressure <<endl;     
 // cout<< "count= "<< clock <<" f= "<<sum_f/NP <<" pressure="<<sum_pressure <<endl;
      clock = 0;
      break;
    }
    
  }
  file.close();
  
  cudaFree(x_dev);
  cudaFree(vx_dev);
  cudaFree(y_dev);
  cudaFree(vy_dev);
  cudaFree(dx_dev);
  cudaFree(dy_dev);
  cudaFree(pot_dev);
  cudaFree(gate_dev);
  cudaFree(L_dev);
  cudaFree(state);
  free(x); 
  free(vx); 
  free(y); 
  free(vy);
  free(pot); 
  return 0;
}

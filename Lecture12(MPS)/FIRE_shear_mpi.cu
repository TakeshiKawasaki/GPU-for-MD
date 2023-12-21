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
#include <mpi.h>
using namespace std;

//Using "const", the variable is shared into both gpu and cpu. 
const int  NT = 512; //Num of the cuda threads.
const int  NP = 9728; //Particle number.
const int  NB = (NP+NT-1)/NT; //Num of the cuda blocks.
const int  NN = 100;
const int  NPC = 100; // Number of the particles in the neighbour cell 
const double dt0 = 0.005;
const double dtmax = 0.05;
const double dtmin = 0.005;
const double RCHK = 2.2;
const double rcut = 1.4;
const double phiini = 0.839;
const double phimax = 0.845;
const double f_thresh = 1.e-12;
const double deltaphi_observe = 1.e-4;
const int ens_ini = 1;

//Initialization of "curandState"
__global__ void setCurand(unsigned long long seed, curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  curand_init(seed, i_global, 0, &state[i_global]);
}

__global__ void eom_kernel(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,double *L_dev,double *dt_dev, int *FIRE_gate_dev,double *gamma_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double y_temp;
  
  if(i_global<NP){
    vx_dev[i_global] +=  fx_dev[i_global]*dt_dev[0];
    vy_dev[i_global] +=  fy_dev[i_global]*dt_dev[0];
    x_dev[i_global]  +=  vx_dev[i_global]*dt_dev[0];
    y_dev[i_global]  +=  vy_dev[i_global]*dt_dev[0];
    y_temp = y_dev[i_global];
    y_dev[i_global]  -= (*L_dev)*floor(y_dev[i_global]/(*L_dev));
    x_dev[i_global]  -= *gamma_dev*(*L_dev)*floor(y_temp/(*L_dev));
    x_dev[i_global]  -= (*L_dev)*floor((x_dev[i_global]-(*gamma_dev)*y_dev[i_global])/(*L_dev));
  }
  if(i_global == 0)
    FIRE_gate_dev[0] = 1;  
  __syncthreads();
}


__global__ void eom_kernel_SS(double*x_dev,double*y_dev,double *vx_dev,double *vy_dev,double *fx_dev,double *fy_dev,double *L_dev,double *dt_dev, int *FIRE_gate_dev,double *gamma_dev,double *gamma_dot_dev,double *stress_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double y_temp;

  if(i_global<NP){
    vx_dev[i_global] +=  fx_dev[i_global]*dt_dev[0];
    vy_dev[i_global] +=  fy_dev[i_global]*dt_dev[0];
    x_dev[i_global]  +=  vx_dev[i_global]*dt_dev[0];
    y_dev[i_global]  +=  vy_dev[i_global]*dt_dev[0];
   
    x_dev[i_global]  +=  *gamma_dot_dev*y_dev[i_global]*dt_dev[0];

    y_temp = y_dev[i_global];
    y_dev[i_global]  -= (*L_dev)*floor(y_dev[i_global]/(*L_dev));
    x_dev[i_global]  -= *gamma_dev*(*L_dev)*floor(y_temp/(*L_dev));
    x_dev[i_global]  -= (*L_dev)*floor((x_dev[i_global]-(*gamma_dev)*y_dev[i_global])/(*L_dev));
  }
  if(i_global == 0){
    FIRE_gate_dev[0] = 1;
    *gamma_dot_dev -= (*L_dev)*stress_dev[0]*(dt_dev[0]);
    *gamma_dev += (*gamma_dot_dev)*(dt_dev[0]);
    //  printf("stress=%.20f,gamma= %.20f,x=%.12f\n",stress_dev[0],*gamma_dot_dev,x_dev[0]);
  }
  __syncthreads();
}



__global__ void FIRE_synth_dev(double *vx_dev,double *vy_dev, double *fx_dev, double *fy_dev, double *power_dev,double *alpha_dev,int *FIRE_gate_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double f,v;
  if(i_global<NP){
    f = sqrt(fx_dev[i_global]*fx_dev[i_global]+fy_dev[i_global]*fy_dev[i_global]);
    v = sqrt(vx_dev[i_global]*vx_dev[i_global]+vy_dev[i_global]*vy_dev[i_global]);
    vx_dev[i_global] = (1.-alpha_dev[0])*vx_dev[i_global]+alpha_dev[0]*v*fx_dev[i_global]/(f+DBL_EPSILON);
    vy_dev[i_global] = (1.-alpha_dev[0])*vy_dev[i_global]+alpha_dev[0]*v*fy_dev[i_global]/(f+DBL_EPSILON);
    power_dev[i_global] = vx_dev[i_global]*fx_dev[i_global]+vy_dev[i_global]*fy_dev[i_global];
    if(f > f_thresh){
      FIRE_gate_dev[0]=0;      
    }
  }
  __syncthreads();
  //if(i_global==0)
  // printf("f=%.25f,fx=%.25f,fy=%.25f\n",f,fx_dev[0],fy_dev[0]);
}




__global__ void FIRE_synth_dev_SS(double *vx_dev,double *vy_dev, double *fx_dev, double *fy_dev, double *power_dev,double *alpha_dev,int *FIRE_gate_dev,double *stress_dev,double *gamma_dot_dev,double *L_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  double f,v;
  if(i_global<NP){
    f = sqrt(fx_dev[i_global]*fx_dev[i_global]+fy_dev[i_global]*fy_dev[i_global]);
    v = sqrt(vx_dev[i_global]*vx_dev[i_global]+vy_dev[i_global]*vy_dev[i_global]);
    vx_dev[i_global] = (1.-alpha_dev[0])*vx_dev[i_global]+alpha_dev[0]*v*fx_dev[i_global]/(f+DBL_EPSILON);
    vy_dev[i_global] = (1.-alpha_dev[0])*vy_dev[i_global]+alpha_dev[0]*v*fy_dev[i_global]/(f+DBL_EPSILON);
    // gdot=(1-alpha)*gdot-alpha*rfxy/sqrt(rfxy*rfxy+DBL_EPSILON)*sqrt(gdot*gdot);
    *gamma_dot_dev =  (1.-alpha_dev[0])**gamma_dot_dev - alpha_dev[0]*stress_dev[0]/(sqrt(stress_dev[0]*stress_dev[0])+DBL_EPSILON)*sqrt(*gamma_dot_dev**gamma_dot_dev);
     power_dev[i_global] = vx_dev[i_global]*fx_dev[i_global]+vy_dev[i_global]*fy_dev[i_global];

    if(f > f_thresh){
      FIRE_gate_dev[0]=0;      
    }
  }
  //  if(i_global == 0 && sqrt(stress_dev[0]*stress_dev[0])**L_dev > f_thresh)
  //   FIRE_gate_dev[0]=0;
  __syncthreads();
  //if(i_global==0)
  // printf("f=%.25f,fx=%.25f,fy=%.25f\n",f,fx_dev[0],fy_dev[0]);
}


__global__ void FIRE_reset_dev_SS(double *vx_dev, double *vy_dev,double *power_dev,double *alpha_dev,double *dt_dev,int *FIRE_param_gate_dev,double *gamma_dot_dev, double *stress_dev, double *L_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;

  //  if(i_global ==0)
  //  printf("device= %.12f\n",power_dev[0]);

  if(i_global<NP){
    if(power_dev[0] + sqrt(stress_dev[0]*stress_dev[0])**gamma_dot_dev*(*L_dev)*(*L_dev) < 0){
      vx_dev[i_global] = 0.0; vy_dev[i_global] = 0.0;
      if(i_global == 0){
	*gamma_dot_dev=0.0;
	alpha_dev[0] = 0.1;
	if(dt_dev[0] > dtmin)
	  dt_dev[0] *= 0.5;
	FIRE_param_gate_dev[0]=0;
	// printf("reset dt=%f\n",dt_dev[0]);
      }
    }
    else{ //five-times criterion
      if(i_global == 0){
	FIRE_param_gate_dev[0]++;
	if(FIRE_param_gate_dev[0]>4){
	  //printf("power=%.25f,alpha=%.16f,dt=%f\n",power_dev[0],alpha_dev[0],dt_dev[0]);
	  alpha_dev[0] *= 0.99;
	  if(dt_dev[0] < dtmax)
	    dt_dev[0] *= 1.1;
	  FIRE_param_gate_dev[0]=0;
	}
      }
    }
  }
  __syncthreads();
}


__global__ void FIRE_reset_dev(double *vx_dev, double *vy_dev,double *power_dev,double *alpha_dev,double *dt_dev,int *FIRE_param_gate_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;

  //  if(i_global ==0)
  //  printf("device= %.12f\n",power_dev[0]);

  if(i_global<NP){
    if(power_dev[0] < 0){
      vx_dev[i_global] = 0.0; vy_dev[i_global] = 0.0;
      if(i_global == 0){
	alpha_dev[0] = 0.1;
	if(dt_dev[0] > dtmin)
	  dt_dev[0] *= 0.5;
	FIRE_param_gate_dev[0]=0;
	// printf("reset dt=%f\n",dt_dev[0]);
      }
    }
    else{ //five-times criterion
      if(i_global ==0){
	FIRE_param_gate_dev[0]++;
	if(FIRE_param_gate_dev[0]>4){
	  //printf("power=%.25f,alpha=%.16f,dt=%f\n",power_dev[0],alpha_dev[0],dt_dev[0]);
	  alpha_dev[0] *= 0.99;
	  if(dt_dev[0] < dtmax)
	    dt_dev[0] *= 1.1;
	  FIRE_param_gate_dev[0]=0;
	}
      }
    }
  }
  __syncthreads();
}



__global__ void disp_gate_kernel(double *vx_dev,double *vy_dev,double *dx_dev,double *dy_dev,int *gate_dev,double *dt_dev)
{
  double r2;  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(i_global<NP){
    dx_dev[i_global]+=vx_dev[i_global]*dt_dev[0];
    dy_dev[i_global]+=vy_dev[i_global]*dt_dev[0];
    r2 = dx_dev[i_global]*dx_dev[i_global]+dy_dev[i_global]*dy_dev[i_global];
    if(r2> 0.05*(RCHK-rcut)*(RCHK-rcut)){
      gate_dev[0]=1;
    }
  }
  __syncthreads();
}


__global__ void update(double *L_dev,double *x_dev,double *y_dev,double *dx_dev,double *dy_dev,int *list_dev,int *gate_dev,double *gamma_dev)
{
  double dx,dy,dy_temp,r2;  
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  
  if(gate_dev[0] == 1 && i_global<NP){
    list_dev[NN*i_global]=0;      
    for (int j=0; j<NP; j++)
      if(j != i_global){
      	dx = x_dev[i_global] - x_dev[j];
	dy = y_dev[i_global] - y_dev[j];
	dy_temp=dy;
	dy -=(*L_dev)*floor(dy/(*L_dev)+0.5);
	dx -= *gamma_dev*(*L_dev)*floor((dy_temp+0.5*(*L_dev))/(*L_dev));
	dx -=(*L_dev)*floor(dx/(*L_dev)+0.5);
	r2 = dx*dx + dy*dy;
	if(r2 < RCHK*RCHK){
	  list_dev[NN*i_global]++;
	  list_dev[NN*i_global+list_dev[NN*i_global]]=j;
	}
      }
    // printf("i=%d, list=%d\n",i_global,list_dev[NN*i_global]);      
    dx_dev[i_global]=0.;
    dy_dev[i_global]=0.;
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
    if(i_global == 0)
       printf("i=%d, list=%d\n",i_global,list_dev[NN*i_global]);      
    dx_dev[i_global]=0.;
    dy_dev[i_global]=0.;
  }
  __syncthreads();
  if(i_global == 0)
    gate_dev[0] = 0;
}

__global__ void calc_force_kernel(double*x_dev,double*y_dev,double *fx_dev,double *fy_dev,double *a_dev,double *L_dev,int *list_dev,double *gamma_dev, double *stress_dev){
  double dx,dy,dy_temp,dr,dU_r,a_ij;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    fx_dev[i_global]=0.0;
    fy_dev[i_global]=0.0;
    stress_dev[i_global]=0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){
      dx = x_dev[list_dev[NN*i_global+j]]-x_dev[i_global];
      dy = y_dev[list_dev[NN*i_global+j]]-y_dev[i_global];

      dy_temp=dy;
      dy -=(*L_dev)*floor(dy/(*L_dev)+0.5);
      dx -= *gamma_dev*(*L_dev)*floor((dy_temp+0.5*(*L_dev))/(*L_dev));
      dx -=(*L_dev)*floor(dx/(*L_dev)+0.5);

      dr = sqrt(dx*dx+dy*dy);
      a_ij = 0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);
      if(dr < a_ij){
	dU_r = -(1-dr/a_ij)/a_ij; //derivertive of U wrt r.
	fx_dev[i_global] += dU_r*dx/dr;
	fy_dev[i_global] += dU_r*dy/dr;
	stress_dev[i_global] += 0.5*dx*dy*dU_r/dr/((*L_dev)*(*L_dev));
      }
    }
    //     printf("i=%d, fx=%f\n",i_global,fx_dev[i_global]);
  }
  __syncthreads();
}

__global__ void calc_energy_kernel(double*x_dev,double*y_dev,double *pressure_dev,double *stress_dev,double *pot_dev,double *a_dev,double *L_dev,int *list_dev,double *gamma_dev){
  double dx,dy,dy_temp,dr,a_ij,dU_r;
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    pot_dev[i_global]=0.0;
    stress_dev[i_global]=0.0;
    pressure_dev[i_global]=0.0;
    for(int j = 1; j<=list_dev[NN*i_global]; j++){
      dx = x_dev[list_dev[NN*i_global+j]]-x_dev[i_global];
      dy = y_dev[list_dev[NN*i_global+j]]-y_dev[i_global];

      dy_temp=dy;
      dy -=(*L_dev)*floor(dy/(*L_dev)+0.5);
      dx -= *gamma_dev*(*L_dev)*floor((dy_temp+0.5*(*L_dev))/(*L_dev));
      dx -=(*L_dev)*floor(dx/(*L_dev)+0.5);

      dr = sqrt(dx*dx+dy*dy);
      a_ij= 0.5*(a_dev[i_global]+a_dev[list_dev[NN*i_global+j]]);
      if(dr < a_ij){
	dU_r = -(1-dr/a_ij)/a_ij;
	pressure_dev[i_global] -= 0.25*dU_r*dr/((*L_dev)*(*L_dev));
	stress_dev[i_global] += 0.5*dx*dy*dU_r/dr/((*L_dev)*(*L_dev));
	pot_dev[i_global] += 0.25*(1.-dr/a_ij)*(1.-dr/a_ij);
      }
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
  if(i_global>=NP/2 && i_global<NP)
    a_dev[i_global] = 1.4;
  __syncthreads();
}

__global__ void init_array_rand(double *x_dev, double c,curandState *state){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  x_dev[i_global] = c*curand_uniform(&state[i_global]);
}

__global__ void change_deltaphi(double *deltaphi_dev,double *phi_dev, double *pot_dev){
  if(*phi_dev >= phimax)
    *deltaphi_dev = -1.e-4;
  if(*deltaphi_dev < 0 && pot_dev[0]/NP < 1.e-9)
    *deltaphi_dev = -1.e-5;
  if(*deltaphi_dev < 0 && pot_dev[0]/NP < 1.e-11)
    *deltaphi_dev = -1.e-6;
 
   __syncthreads();
}

__global__ void volume_affine(double *x_dev, double *y_dev,double *phi_dev,double *deltaphi_dev,double *L_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
  if(i_global<NP){
    x_dev[i_global] = x_dev[i_global]*sqrt(*phi_dev/(*phi_dev+*deltaphi_dev));
    y_dev[i_global] = y_dev[i_global]*sqrt(*phi_dev/(*phi_dev+*deltaphi_dev));
  }
  __syncthreads();
  
  
  if(i_global==0){
    *L_dev   *= sqrt(*phi_dev/(*phi_dev+*deltaphi_dev));
    *phi_dev = M_PI*(1.0*1.0+1.4*1.4)*(double)NP/8./(*L_dev)/(*L_dev);
  }
  __syncthreads();
}

__global__ void change_delta_gamma(double *delta_gamma_dev,double *gamma_dev){
  *delta_gamma_dev *=pow(10.,0.1);
  if(*gamma_dev>1.e-3)
    *delta_gamma_dev= 1.e-3;

  __syncthreads();

}


__global__ void shear_affine(double *x_dev, double *y_dev,double *delta_gamma_dev,double *gamma_dev,double *gamma_observe_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;
 
  if(i_global == 0){
    *gamma_dev +=*delta_gamma_dev;
    *gamma_observe_dev +=*delta_gamma_dev;
  }
  
  if(i_global<NP){
    x_dev[i_global] +=y_dev[i_global]*(*delta_gamma_dev);
  }
  __syncthreads();
}


__global__ void shear_affine_SS(double *x_dev, double *y_dev,double *gamma_dot_dev, double *dt_dev){
  int i_global = threadIdx.x + blockIdx.x*blockDim.x;

  if(i_global<NP){
    x_dev[i_global] +=y_dev[i_global]*(*gamma_dot_dev)*dt_dev[0];
  }
  __syncthreads();

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


void list_auto_update(double *L_dev,double *x_dev, double *y_dev,double *vx_dev, double *vy_dev, double *dx_dev,double *dy_dev, double *dt_dev, int M, int *map_dev, int *gate_dev,int *list_dev,double *gamma_dev){
  disp_gate_kernel<<<NB,NT>>>(vx_dev,vy_dev,dx_dev,dy_dev,gate_dev,dt_dev);
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


void FIRE(double *x_dev, double *y_dev,double  *vx_dev, double *vy_dev, double *fx_dev,double *fy_dev, double *a_dev,double *L_dev, int *list_dev, double *power_dev, double *alpha_dev, double *dt_dev, int *FIRE_gate_dev,int *FIRE_param_gate_dev, int *reduce_dev, int *remain_dev,double *gamma_dev,double *stress_dev){

  calc_force_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,L_dev,list_dev,gamma_dev,stress_dev);
  eom_kernel<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,L_dev,dt_dev,FIRE_gate_dev,gamma_dev);
  FIRE_synth_dev<<<NB,NT>>>(vx_dev,vy_dev,fx_dev,fy_dev,power_dev,alpha_dev,FIRE_gate_dev);
  double sum_power = thrust::reduce(thrust::device_pointer_cast(power_dev), thrust::device_pointer_cast(power_dev + NB * NT),0.0,thrust::plus<double>());
  // cout <<"sum(host)=" << sum_power <<endl;
  //  add_reduction_array(power_dev,reduce_dev,remain_dev);
  cudaMemcpy(&power_dev[0], &sum_power, sizeof(double), cudaMemcpyHostToDevice);
  FIRE_reset_dev<<<NB,NT>>>(vx_dev,vy_dev,power_dev,alpha_dev,dt_dev,FIRE_param_gate_dev); 
}


void FIRE_SS(double *x_dev, double *y_dev,double  *vx_dev, double *vy_dev, double *fx_dev,double *fy_dev, double *a_dev,double *L_dev, int *list_dev, double *power_dev, double *alpha_dev, double *dt_dev, int *FIRE_gate_dev,int *FIRE_param_gate_dev, int *reduce_dev, int *remain_dev,double *gamma_dev,double *gamma_dot_dev, double *stress_dev){

  calc_force_kernel<<<NB,NT>>>(x_dev,y_dev,fx_dev,fy_dev,a_dev,L_dev,list_dev,gamma_dev,stress_dev);
  double sum_stress = thrust::reduce(thrust::device_pointer_cast(stress_dev), thrust::device_pointer_cast(stress_dev + NB * NT),0.0,thrust::plus<double>());
  cudaMemcpy(&stress_dev[0], &sum_stress, sizeof(double), cudaMemcpyHostToDevice);
 
  eom_kernel_SS<<<NB,NT>>>(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,L_dev,dt_dev,FIRE_gate_dev,gamma_dev,gamma_dot_dev,stress_dev);
  FIRE_synth_dev_SS<<<NB,NT>>>(vx_dev,vy_dev,fx_dev,fy_dev,power_dev,alpha_dev,FIRE_gate_dev,stress_dev,gamma_dot_dev,L_dev);

  double sum_power = thrust::reduce(thrust::device_pointer_cast(power_dev), thrust::device_pointer_cast(power_dev + NB * NT),0.0,thrust::plus<double>());
  cudaMemcpy(&power_dev[0], &sum_power, sizeof(double), cudaMemcpyHostToDevice);
  FIRE_reset_dev_SS<<<NB,NT>>>(vx_dev,vy_dev,power_dev,alpha_dev,dt_dev,FIRE_param_gate_dev,gamma_dot_dev,stress_dev,L_dev);
  // shear_affine_SS<<<NB,NT>>>(x_dev,y_dev,gamma_dot_dev,dt_dev);
}


void output(double *x,double *y,double *a){
  static int count=1;
  char filename[128];
  sprintf(filename,"coord_%d.dat",count);
  ofstream file;
  file.open(filename);
  
  for(int i=0;i<NP;i++)
    file << x[i] << " " << y[i]<< " " << a[i] << endl;
  
  file.close();
  count++;
}



int main(int argc, char** argv){
  double *x,*vx,*y,*vy,*x_dev,*vx_dev,*y_dev,*dx_dev,*dy_dev,*vy_dev,*a,*a_dev,*fx_dev,*fy_dev,*power_dev,*L_dev,*deltaphi_dev,deltaphi;
  double *pot_dev,*pressure_dev,*stress_dev,*pot,*pressure,*stress;
  double *dt_dev,*alpha_dev,*phi_dev,phi;
  double gamma =0.,*gamma_dev,*gamma_observe_dev,*delta_gamma_dev,*gamma_dot_dev;
  int *list_dev,*map_dev,*gate_dev,*remain_dev,*reduce_dev;
  int *FIRE_gate_dev,FIRE_gate,*FIRE_param_gate_dev;
  int clock=0;
  char filename[128];
 
  curandState *state; //Cuda state for random numbers
  double sec; //measurred time
  double L = sqrt(M_PI*(1.0*1.0+1.4*1.4)*(double)NP/(8.*phiini));
  int M = (int)(L/RCHK);

  int np,myrank;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  int gpu_id = myrank;
  cudaSetDevice(gpu_id); 

  cout <<" M= "<< M <<" L= "<<L<<endl;



  
  x  = (double*)malloc(NB*NT*sizeof(double));
  y  = (double*)malloc(NB*NT*sizeof(double));
  vx = (double*)malloc(NB*NT*sizeof(double));
  vy = (double*)malloc(NB*NT*sizeof(double));
  a  = (double*)malloc(NB*NT*sizeof(double));
  pot  = (double*)malloc(NB*NT*sizeof(double));
  stress  = (double*)malloc(NB*NT*sizeof(double));
  pressure  = (double*)malloc(NB*NT*sizeof(double));
  cudaMalloc((void**)&x_dev,  NB * NT * sizeof(double)); // CudaMalloc should be executed once
  cudaMalloc((void**)&y_dev,  NB * NT * sizeof(double));
  cudaMalloc((void**)&dx_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&dy_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&vy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&fx_dev, NB * NT * sizeof(double)); 
  cudaMalloc((void**)&fy_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&pot_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&stress_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&pressure_dev, NB * NT * sizeof(double));
  cudaMalloc((void**)&a_dev,  NB * NT * sizeof(double)); 
  cudaMalloc((void**)&power_dev,  NB * NT * sizeof(double));
  cudaMalloc((void**)&dt_dev, sizeof(double)); 
  cudaMalloc((void**)&alpha_dev, sizeof(double));
  cudaMalloc((void**)&L_dev, sizeof(double)); 
  cudaMalloc((void**)&phi_dev, sizeof(double));
  cudaMalloc((void**)&deltaphi_dev, sizeof(double)); 
  cudaMalloc((void**)&delta_gamma_dev, sizeof(double));
  cudaMalloc((void**)&gamma_dev,sizeof(double));
  cudaMalloc((void**)&gamma_dot_dev,sizeof(double));
  cudaMalloc((void**)&gamma_observe_dev,sizeof(double));  
  cudaMalloc((void**)&gate_dev, sizeof(int));
  cudaMalloc((void**)&FIRE_gate_dev, sizeof(int)); 
  cudaMalloc((void**)&FIRE_param_gate_dev, sizeof(int)); 
  cudaMalloc((void**)&remain_dev, sizeof(int));
  cudaMalloc((void**)&reduce_dev, sizeof(int));
  cudaMalloc((void**)&list_dev,  NB * NT * NN* sizeof(int)); 
  cudaMalloc((void**)&map_dev,  M * M * NPC* sizeof(int)); 
  cudaMalloc((void**)&state,  NB * NT * sizeof(curandState)); 
 




  sprintf(filename,"stress_phimax%f_dphi%f_ens%d.dat",phimax,deltaphi_observe,gpu_id);
  ofstream file;
  file.open(filename);
  
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
  init_gate_kernel<<<1,1>>>(FIRE_gate_dev,0);
  init_gate_kernel<<<1,1>>>(FIRE_param_gate_dev,0);
  init_scalar_kernel<<<1,1>>>(dt_dev,dt0);
  init_scalar_kernel<<<1,1>>>(alpha_dev,0.1);
  init_scalar_kernel<<<1,1>>>(gamma_dot_dev,0.0);
  init_map_kernel<<<M*M,NPC>>>(map_dev,M);
  cell_map<<<NB,NT>>>(L_dev,x_dev,y_dev,map_dev,gate_dev,M,gamma_dev);
  cell_list<<<NB,NT>>>(L_dev,x_dev,y_dev,dx_dev,dy_dev,list_dev,map_dev,gate_dev,M,gamma_dev);  
  
  init_scalar_kernel<<<1,1>>>(deltaphi_dev,1.e-4);
  init_scalar_kernel<<<1,1>>>(gamma_dev,0.0);
  init_scalar_kernel<<<1,1>>>(gamma_observe_dev,0.0);
  init_scalar_kernel<<<1,1>>>(gamma_dot_dev,0.0);
  cudaMemcpy(a,a_dev,NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
  
  
  ///Random to the relaxed: without shear stabilization.
  for(;;){
    FIRE(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,a_dev,L_dev,list_dev,power_dev,alpha_dev,dt_dev,FIRE_gate_dev,FIRE_param_gate_dev,reduce_dev,remain_dev,gamma_dev,stress_dev);                                            
    list_auto_update(L_dev,x_dev,y_dev,vx_dev,vy_dev,dx_dev,dy_dev,dt_dev,M,map_dev,gate_dev,list_dev,gamma_dev);
    cudaMemcpy(&FIRE_gate,FIRE_gate_dev,sizeof(int),cudaMemcpyDeviceToHost);
    if(FIRE_gate == 1)
	break;
    }
    
    ///Compression and Decompression with the shear stabilization to remove the redidual stress. 
    for(;;){
      measureTime();  
      for(;;){
	clock++;
	FIRE_SS(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,a_dev,L_dev,list_dev,power_dev,alpha_dev,dt_dev,FIRE_gate_dev,FIRE_param_gate_dev,reduce_dev,remain_dev,gamma_dev,gamma_dot_dev,stress_dev);
	list_auto_update(L_dev,x_dev,y_dev,vx_dev,vy_dev,dx_dev,dy_dev,dt_dev,M,map_dev,gate_dev,list_dev,gamma_dev);
	cudaMemcpy(&FIRE_gate,FIRE_gate_dev,sizeof(int),cudaMemcpyDeviceToHost);
	if(FIRE_gate == 1){
	  calc_energy_kernel<<<NB,NT>>>(x_dev,y_dev,pressure_dev,stress_dev,pot_dev,a_dev,L_dev,list_dev,gamma_dev);
	  add_reduction_array(pot_dev,reduce_dev,remain_dev);
	  cudaMemcpy(pot,pot_dev,NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(&phi,phi_dev, sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(&deltaphi,deltaphi_dev, sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(stress,stress_dev, NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(&gamma,gamma_dev, sizeof(double),cudaMemcpyDeviceToHost);
	  cout<<"phi= "<< phi << " count= "<< clock <<" pot= "<<pot[0]/NP  <<" stress= "<< stress[0] <<" gamma= "<< gamma   <<endl;
	  clock = 0;
	  break;
	}
	//////////////////////////
      }
      if(pot[0]/NP < 1.e-14 && deltaphi<0){
	cout<<" phiJ= "<< phi <<endl;
	break;
      }
      change_deltaphi<<<1,1>>>(deltaphi_dev,phi_dev,pot_dev);
      volume_affine<<<NB,NT>>>(x_dev,y_dev,phi_dev,deltaphi_dev,L_dev);
      sec = measureTime()/1000.;
      cout<<"time(sec):"<< sec <<endl;
    }
    
    double phiJ = phi;
    init_scalar_kernel<<<1,1>>>(deltaphi_dev,1.e-5);
    
    for(;;){
      measureTime();  
      for(;;){
	clock++;
	FIRE_SS(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,a_dev,L_dev,list_dev,power_dev,alpha_dev,dt_dev,FIRE_gate_dev,FIRE_param_gate_dev,reduce_dev,remain_dev,gamma_dev,gamma_dot_dev,stress_dev);
	list_auto_update(L_dev,x_dev,y_dev,vx_dev,vy_dev,dx_dev,dy_dev,dt_dev,M,map_dev,gate_dev,list_dev,gamma_dev);
	cudaMemcpy(&FIRE_gate,FIRE_gate_dev,sizeof(int),cudaMemcpyDeviceToHost);
	
	if(FIRE_gate == 1){
	  calc_energy_kernel<<<NB,NT>>>(x_dev,y_dev,pressure_dev,stress_dev,pot_dev,a_dev,L_dev,list_dev,gamma_dev);
	  add_reduction_array(pot_dev,reduce_dev,remain_dev);
	  cudaMemcpy(pot,pot_dev,NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(&phi,phi_dev, sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(&deltaphi,deltaphi_dev, sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(stress,stress_dev, NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(&gamma,gamma_dev, sizeof(double),cudaMemcpyDeviceToHost);
	  cout<<"phi= "<< phi << " count= "<< clock <<" pot= "<<pot[0]/NP  <<" stress= "<< stress[0] <<" gamma= "<< gamma   <<endl;
	  clock = 0;
	  break;
	}
	//////////////////////////
      }
      if(phi >= phiJ + deltaphi_observe){
	break;
      }
      change_deltaphi<<<1,1>>>(deltaphi_dev,phi_dev,pot_dev);
      volume_affine<<<NB,NT>>>(x_dev,y_dev,phi_dev,deltaphi_dev,L_dev);
      sec = measureTime()/1000.;
      cout<<"time(sec):"<< sec <<endl;
    }
    
    ///initiallization/////////////////
    init_scalar_kernel<<<1,1>>>(delta_gamma_dev, 1.e-7);
    
    file<< "# gamma stress[0] pressure[0]"<<endl;       
    file<< phiJ <<endl;
    ///shear///////////////
    while(gamma < 0.1){ 
      shear_affine<<<NB,NT>>>(x_dev,y_dev,delta_gamma_dev,gamma_dev,gamma_observe_dev);
      for(;;){
	clock++;
	FIRE(x_dev,y_dev,vx_dev,vy_dev,fx_dev,fy_dev,a_dev,L_dev,list_dev,power_dev,alpha_dev,dt_dev,FIRE_gate_dev,FIRE_param_gate_dev,reduce_dev,remain_dev,gamma_dev,stress_dev);
	list_auto_update(L_dev,x_dev,y_dev,vx_dev,vy_dev,dx_dev,dy_dev,dt_dev,M,map_dev,gate_dev,list_dev,gamma_dev);   
	cudaMemcpy(&FIRE_gate,FIRE_gate_dev,sizeof(int),cudaMemcpyDeviceToHost);
	if(FIRE_gate == 1){
	  calc_energy_kernel<<<NB,NT>>>(x_dev,y_dev,pressure_dev,stress_dev,pot_dev,a_dev,L_dev,list_dev,gamma_dev);
	  add_reduction_array(stress_dev,reduce_dev,remain_dev);
	  add_reduction_array(pressure_dev,reduce_dev,remain_dev);
	  cudaMemcpy(stress,stress_dev,NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(pressure,pressure_dev,NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(&gamma,gamma_observe_dev, sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(x,x_dev,NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  cudaMemcpy(y,y_dev,NB*NT*sizeof(double),cudaMemcpyDeviceToHost);
	  
	  cout<<"gamma= "<< gamma << " count= "<< clock <<" stress= "<<stress[0] <<" pressure= "<< pressure[0] <<endl;
	  file<< gamma<<" "<<stress[0]<<" "<< pressure[0] <<endl;
	  clock = 0;
	  break;
	}
      }
      output(x,y,a);
      change_delta_gamma<<<1,1>>>(delta_gamma_dev,gamma_observe_dev);
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
  cudaFree(dt_dev);
  cudaFree(alpha_dev);
  cudaFree(L_dev);
  cudaFree(state);
  free(x); 
  free(vx); 
  free(y); 
  free(vy);
  free(pot); 
  return 0;
}

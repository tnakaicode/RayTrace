#include <cstdio>
#include <cassert>
#include <math.h>
#include "rhs.h"
//
using namespace std;

/* For every problem (e.g., harmonic oscillator, particle_in_tmag ) 
create two files, one header file 'harmonic.h', and a corresponding
.cu file e.g, harmonic.cu. The main part of the code should be
gode.cu and the ode solver should be evolve.cu which would contain
the function rnkt4 (or rnkt2 or any other) */

bool lyapunov = false;
__constant__ R kk = 32.;
__constant__ R qbym = 1.;
__constant__ int bfield = 1;
__constant__ R mag_BB = 1.;
__constant__ R mag_BBc = 1.;

__device__
void crossmultiply(R *x, R *y, R *result){
  result[0] = x[1]*y[2] - x[2]*y[1];
  result[1] = -(x[0]*y[2]) + x[2]*y[0];
  result[2] = x[0]*y[1] - x[1]*y[0];
}

__device__
void dotmultiply(R x[3][3], R *y, R *result){
  result[0] = (x[0][0]) * (y[0]) + (x[1][0]) * (y[1]) + (x[2][0]) * (y[2]);
  result[1] = x[0][1] * y[0] + x[1][1] * y[1] + x[2][1] * y[2];
  result[2] = x[0][2] * y[0] + x[1][2] * y[1] + x[2][2] * y[2];
}

__device__
void ABC(R *x, R *magfield, R dmagfield[3][3]){
  magfield[0] = cos(kk * x[1]) + sin(kk * x[2]);
  magfield[1] = sin(kk * x[0]) + cos(kk * x[2]);
  magfield[2] = sin(kk * x[2]) + cos(kk * x[0]);

/*  dmagfield[0][0] = kk * 0;
  dmagfield[0][1] = -kk * sin(kk * x[1]);
  dmagfield[0][2] = kk * cos(kk * x[2]);
  dmagfield[1][0] = kk * cos(kk * x[0]);
  dmagfield[1][1] = kk * 0;
  dmagfield[1][2] = -kk * sin(kk * x[2]);
  dmagfield[2][0] = -kk * sin(kk * x[0]);
  dmagfield[2][1] = kk * cos(kk * x[1]);
  dmagfield[2][2] = kk * 0; */
}
__device__
void ABC_constz(R *x, R *magfield, R dmagfield[3][3]){
  R Bzero=1.;
  R epsB=0.;
  ABC( x,  magfield,  dmagfield);
  for(int j=0;j<3;j++){
    magfield[j] = epsB*magfield[j];
  }
  magfield[2]=magfield[2]+Bzero;
}
__device__
void constB(R *magfield, R dmagfield[3][3]){
  magfield[0] = 0.;
  magfield[1] = 0.;
  magfield[2] = 1.;

  dmagfield[0][0] = 0;
  dmagfield[0][1] = 0.;
  dmagfield[0][2] = 0.;
  dmagfield[1][0] = 0.;
  dmagfield[1][1] = 0;
  dmagfield[1][2] = 0.;
  dmagfield[2][0] = 0. ;
  dmagfield[2][1] = 0.;
  dmagfield[2][2] = 0;
}

__device__
void genmagfield(R *x, R *magfield, R dmagfield[3][3]){
  ABC_constz(x, magfield, dmagfield);
  //ABC(x, magfield, dmagfield);
  //constB(magfield,dmagfield);
}

__host__
void init_values(R *y, R *init_y_values, unsigned long int nx, int ndim) {
  for(int i=0;i<nx;i++) {
    y[i*ndim+0]=init_y_values[i*ndim+0] = -1.0 + (R)rand()/((R)RAND_MAX/2.0);
    y[i*ndim+1]=init_y_values[i*ndim+1] = -1.0 + (R)rand()/((R)RAND_MAX/2.0);
    y[i*ndim+2]=init_y_values[i*ndim+2] = -1.0 + (R)rand()/((R)RAND_MAX/2.0);
    y[i*ndim+3]=init_y_values[i*ndim+3] = 1 + (R)rand()/((R)RAND_MAX/1);
    y[i*ndim+4]=init_y_values[i*ndim+4] = 1 + (R)rand()/((R)RAND_MAX/1);
    y[i*ndim+5]=init_y_values[i*ndim+5] = 1 + (R)rand()/((R)RAND_MAX/1);
//    printf("NX: %d and %f  %f  %f  :  %f  %f  %f\n", i, y[i*ndim+0], y[i*ndim+1], y[i*ndim+2], y[i*ndim+3], y[i*ndim+4], y[i*ndim+5]);
  /* y[i*ndim+0]=1.;
   y[i*ndim+1]=0.;
   y[i*ndim+2]=0.;
   y[i*ndim+3]=0.;
   y[i*ndim+4]=0.;
   y[i*ndim+5]=1.;
   for (int j=0;j<ndim;j++){
   init_y_values[i*ndim+j]=y[i*ndim+j];
   }*/
  }  
}

__host__ 
void modsqr(R *array, R *result, int length) {
  for(int i=0;i<length;i++) {
    *result = *result + (array[i] * array[i]);
  }
}
__host__
void diagnostics(R *y, R *init_y_values, unsigned long int nx, int ndim, R time, FILE *fdiag ) {
  R energy=0;
  R vel[3];
  R r2[3];
  R disp[3];
  for(int ispace=0;ispace<=3;ispace++){
    r2[ispace]=0.;
  }
  for(int i=0;i<nx;i++) {

    //Calculate energy
    vel[0]=y[i*ndim+3];
    vel[1]=y[i*ndim+4];
    vel[2]=y[i*ndim+5];
    modsqr(vel, &energy, 3);

    //Calculate r^2
    disp[0] = y[i*ndim+0] - init_y_values[i*ndim+0];
    disp[1] = y[i*ndim+1] - init_y_values[i*ndim+1];
    disp[2] = y[i*ndim+2] - init_y_values[i*ndim+2];
    for(int ispace=0;ispace<=3;ispace++){
      r2[ispace]=r2[ispace]+disp[ispace]*disp[ispace];
    }
    //modsqr(disp, &r, 3); //Only 2 dimensions
  } 
/*  printf("Current energy is %16.8f!\n", energy);
  printf("Current r2(x,y,z) is %16.8f,%16.8f,%16.8f!\n", r2[0]/(R) nx,r2[1]/(R) nx, r2[2]/ (R) nx);
  printf("%f\t%f\t%f\n", y[0],y[1],y[2]);
  printf("%f\t%f\t%f\n", y[3],y[4],y[5]); */
  fprintf(fdiag, "%32.16f\t%16.8f\t%16.8f\t%16.8f\t%16.8f\n",time,energy/ (R) nx,r2[0]/(R) nx,r2[1]/(R) nx, r2[2]/ (R) nx);
}
__device__
void eval_rhs(R *f, R *df, R time, int istep) {
  bool lyapunov = false;
  R magfield[3],vXB[3]; 
  R dmagfield[3][3];
  df[0] = f[3];
  df[1] = f[4];
  df[2] = f[5];
  genmagfield(&f[0], magfield, dmagfield);
  crossmultiply(&f[3], magfield, vXB);
  df[3] = qbym * vXB[0];
  df[4] = qbym * vXB[1];
  df[5] = qbym * vXB[2];
}

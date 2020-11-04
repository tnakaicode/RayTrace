/*
 * Very basic CUDA application.
 * RK4 integration of the simple harmonic oscillator.
 * Initial condition: sine wave.
 *
 * Use the provided makefile to compile and 
 * follow the instructions to run and visualise the results.
 *
 * Copyright(c) 2013 by Emanuel Gafton
 */
#include <cstdio>
#include <cassert>
using namespace std;

const int ndim       = 2;
const int nx         = 512;
const int nthreads   = 64;
const float omega    = 1.;
const float omegasq  = omega*omega;
const float pi       = 3.1415926;

__device__
inline void eval_rhs(float *f, float *df) {
  assert ( ndim == 2 );
  df[0] = f[1];
  df[1] = -omegasq * f[0];
}

__device__
inline void upp(float *fin, float *df, float *fout, float *dt) {
  for (int i=0; i<ndim; ++i) {
    fout[i] = fin[i] + df[i] * (*dt);
  }
}

__device__
void rk4(float *f, float *t, float *dt) {
  float f1[ndim], f2[ndim], f3[ndim], df[ndim];
  float k1[ndim], k2[ndim], k3[ndim], k4[ndim];
  float dthalf = .5 * (*dt);

  // First step
  eval_rhs(f, k1);
  upp(f, k1, f1, &dthalf);

  // Second step
  eval_rhs(f1, k2);
  upp(f, k2, f2, &dthalf);

  // Third step
  eval_rhs(f2, k3);
  upp(f, k3, f3, dt);

  // Fourth step
  eval_rhs(f3, k4);

  // Calculate derivative
  for (int i=0;i<ndim;++i) {    
    df[i] = (k1[i]/6. + k2[i]/3. + k3[i]/3. + k4[i]/6.);
  }

  // Advance
  upp(f, df, f, dt);
}

__global__
void integrate(float *f, float *t, float *dt) {
  __shared__ float temp[nthreads * ndim];
  int gindex = (threadIdx.x + blockIdx.x * blockDim.x) * ndim;
  int lindex = threadIdx.x * ndim;

  // Read input elements into shared memory
  for(int i=0; i<ndim; ++i) {
    temp[lindex+i] = f[gindex+i];
  }

  // Synchronize threads
  __syncthreads();

  // Perform the RK4 integration
  rk4(&temp[lindex], t, dt);

  // Write the data back
  for(int i=0; i<ndim; ++i) {
    f[gindex+i] = temp[lindex+i];
  }
}

__host__
void cudaReport(void) {
  // Get information about the device
  int dev;
  cudaGetDeviceCount(&dev);
  printf("Number of devices: %d\n",dev);
  cudaGetDevice(&dev);
  printf("Using device:      %d\n",dev);

  cudaDeviceProp devProp;
  cudaGetDeviceProperties(&devProp, dev);
  printf("Device properties:\n");
  printf("\tName:                    %s\n",devProp.name);
  printf("\tTotal global memory:     %lu MB\n",devProp.totalGlobalMem/1024/1024);
  printf("\tShared memory per block: %lu kB\n",devProp.sharedMemPerBlock/1024);
  printf("\tRegisters per block:     %d\n",devProp.regsPerBlock);
  printf("\tWarp size in threads:    %d\n",devProp.warpSize);
  printf("\tMax threads per block:   %d\n",devProp.maxThreadsPerBlock);
  printf("\tMax sizes of a block:    %d %d %d\n",devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
  printf("\tMax sizes of a grid:     %d %d %d\n",devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
  printf("\tTotal constant memory:   %lu kB\n",devProp.totalConstMem/1024);
  printf("\tClock frequency:         %d GHz\n",devProp.clockRate/1024/1024);
  printf("\tTexture alignment:       %lu B\n",devProp.textureAlignment);
  printf("\tDevice overlap:          %s\n",devProp.deviceOverlap?"True":"False");
  printf("\tNo. of multiprocessors:  %d\n",devProp.multiProcessorCount);
  printf("\n");
}

__host__ 
int main(void) {
  // Report CUDA capabilities
  cudaReport();

  // Open results file
  FILE *fout = fopen("z.dat", "w");
  int blocks = 0;

  // Initialize numerical variables
  const int length = nx * ndim;
  printf("Length: %d, i.e. from 0 to %d\n", length, length-1);
  const int fsize = sizeof(float);
  const int psize = length * fsize;
  float y[length], *y_dev;
  float t, dt, *t_dev, *dt_dev;
  t  = 0.;
  dt = .05 * 2.*pi/omega;
  printf("Using dt=%5.2f\n\n",dt);
  for(int i=0;i<nx;i++) {
    y[i*ndim]= sin(2.*pi*(float)i/(float)(nx-1));
    y[i*ndim+1]= 0.;
  }
 
  for(int i=0;i<nx;i++)
    fprintf(fout,"%6.4f\n", y[i*ndim]);
  fprintf(fout,"\n\n");
  ++ blocks;

  // Allocate and initialize device memory
  cudaMalloc( (void**)&y_dev, psize ); 
  cudaMalloc( (void**)&t_dev, fsize );
  cudaMalloc( (void**)&dt_dev, fsize );
  cudaMemcpy(  y_dev,  &y, psize, cudaMemcpyHostToDevice );
  cudaMemcpy( dt_dev, &dt, fsize, cudaMemcpyHostToDevice );
  
  // Define block and grid
  dim3 dimBlock( nthreads, 1 );
  dim3 dimGrid( nx/nthreads, 1 );

  // Now perform the integration
  while (t < 4.*pi/omega) {
    cudaMemcpy(  t_dev,  &t, fsize, cudaMemcpyHostToDevice );

    integrate<<<dimGrid, dimBlock>>>( y_dev, t_dev, dt_dev );
    t += dt;
    cudaMemcpy( &y, y_dev, psize, cudaMemcpyDeviceToHost ); 

    // Print out all intermediate steps
    for(int i=0;i<nx;i++)
      fprintf(fout,"%6.4f\n", y[i*ndim]);
    fprintf(fout,"\n\n");
    ++ blocks;
  }

  // Close file and display final message
  fclose(fout);
  printf("Done. Wrote %d blocks to z.dat.\n", blocks);
  printf("Now run \"gnuplot < plot.gnu\" to view the results.\n");

  cudaFree( y_dev );
	
  return EXIT_SUCCESS;
}

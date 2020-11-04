#include <cstdio>
#include <cassert>
#include "rhs.h"
using namespace std;

/* -------------------------------------- */
/* For every problem (e.g., harmonic oscillator, particle_in_tmag ) 
create two files, one header file 'harmonic.h', and a corresponding
.cu file e.g, harmonic.cu. The main part of the code should be
gode.cu and the ode solver should be evolve.cu which would contain
the function rnkt4 (or rnkt2 or any other) */


__global__
void integrate(R *f, R *t, R *dt, R *tnext);

__host__
void init_values(R *y, R *y0, unsigned long int nx, int ndim);

__host__
void diagnostics(R *y, R *y0, unsigned long int  nx, int ndim, R time, FILE *fdiag);


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
int main(int argc, char *argv[]) {

  clock_t start_time = clock();

  // TMAX should come from and input file, but for now we can take it from command line to simplify things. 
  R TMAX = 10.;
  if ( argc > 1 ) /* argc should be 2 for correct execution */
  {
        /* We print argv[0] assuming it is the program name */
        printf( "debug:: TMAX set by commandline to %s" , argv[1] );
	TMAX = atof(argv[1]);
  }

  // Report CUDA capabilities
  cudaReport();


  // Initialize numerical variables
  const unsigned long int length = nx * ndim;
  printf("Length: %d, i.e. from 0 to %d\n", length, length-1);
  const int fsize = sizeof(R);
  const long int psize = length * fsize;
  R y0[length], y[length], *y_dev;
  R t, dt, *t_dev, *dt_dev;
  R dt_diag, tnext, *tnext_dev; 
  R time_spent; 
  t  = 0.;
  dt = input_dt;
  printf("Using dt=%5.2f\n\n",dt);
  

  init_values(y, y0, nx, ndim);
 
  // Open initial condition file
  FILE *finit = fopen("inity.dat", "w");
  for(int i=0;i<length;i++){
    fprintf(finit,"%32.16f\n", y[i]);
  }
  fclose(finit);

  // Allocate and initialize device memory
  cudaMalloc( (void**)&y_dev, psize ); 
  cudaMalloc( (void**)&t_dev, fsize );
  cudaMalloc( (void**)&dt_dev, fsize );
  cudaMalloc( (void**)&tnext_dev, fsize );
  cudaMemcpy(  y_dev,  &y, psize, cudaMemcpyHostToDevice );
  cudaMemcpy( dt_dev, &dt, fsize, cudaMemcpyHostToDevice );
  cudaMemcpy(  t_dev,  &t, fsize, cudaMemcpyHostToDevice );
  cudaMemcpy(  tnext_dev,  &tnext, fsize, cudaMemcpyHostToDevice );
  
  // Define block and grid
  dim3 dimBlock( nthreads, 1 );
  dim3 dimGrid( nx/nthreads, 1 );
 // Decide how frequently diagnostics will be calculated 
  dt_diag = TMAX/ndiag; 
  // Now perform the integration
  FILE *fdiag = fopen("diag.dat", "w");
  while (t < TMAX ) {
    tnext=t+dt_diag;
    cudaMemcpy(tnext_dev,  &tnext, fsize, cudaMemcpyHostToDevice );
    integrate<<<dimGrid, dimBlock>>>( y_dev, t_dev, dt_dev, tnext_dev );
    cudaMemcpy( &y, y_dev, psize, cudaMemcpyDeviceToHost ); 
    t += dt_diag;
    cudaMemcpy(t_dev,  &t, fsize, cudaMemcpyHostToDevice );
    // Print out all intermediate steps
    printf("Integrated upto, t=%6.4f\n",t);
    diagnostics(y, y0, nx, ndim, t, fdiag);
  }
  fclose(fdiag); 
  clock_t end_time = clock();

  time_spent = (double)(end_time - start_time) / CLOCKS_PER_SEC;
  printf("Elapsed: %f seconds\n", time_spent);
  // Open initial condition file
  FILE *fout = fopen("finity.dat", "w");
  for(int i=0;i<length;i++){
    fprintf(fout,"%32.16f\n", y[i]);
  } 
  fclose(fout);
  // Close file and display final message
  printf("Done. time now is t=%6.4f \n", t);
  printf("Now run \"gnuplot < plot.gnu\" to view the results.\n");

  cudaFree( y_dev );
  cudaFree( t_dev );
  cudaFree( dt_dev );
	
  return EXIT_SUCCESS;
}

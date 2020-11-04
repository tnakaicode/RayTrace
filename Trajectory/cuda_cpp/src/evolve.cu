#include <cstdio>
#include <cassert>
#include "rhs.h" 
using namespace std;

/* For every problem (e.g., harmonic oscillator, particle_in_tmag ) 
create two files, one header file 'harmonic.h', and a corresponding
.cu file e.g, harmonic.cu. The main part of the code should be
gode.cu and the ode solver should be evolve.cu which would contain
the function rnkt4 (or rnkt2 or any other) */
//-------------------
__device__
void eval_rhs(R *f, R *df, R time, int istep);
//----------------------------
__device__
inline void upp(R *fin, R *df, R *fout, R *dt) {
  for (int i=0; i<ndim; ++i) {
    fout[i] = fin[i] + df[i] * (*dt);
  }
}
//-----------------------------
__device__
void rk4(R *f, R *t, R *dt) {
  R f1[ndim], f2[ndim], f3[ndim], df[ndim];
  R k1[ndim], k2[ndim], k3[ndim], k4[ndim];
  R dthalf = .5 * (*dt);
  R time = (*t);
  int istep;
/*
*/

  // First step
  istep=1;
  eval_rhs(f, k1, time, istep);
  upp(f, k1, f1, &dthalf);
  R thalf=time+dthalf; 
  // Second step
  istep=2;
  eval_rhs(f1, k2, thalf, istep);
  upp(f, k2, f2, &dthalf);
  thalf=time+dthalf; 
  // Third step
  istep=3;
  eval_rhs(f2, k3, thalf, istep);
  upp(f, k3, f3, dt);
  thalf=time+(*dt);
  // Fourth step
  istep=4;
  eval_rhs(f3, k4, time, istep);

  // Calculate derivative
  for (int i=0;i<ndim;++i) {    
    df[i] = (k1[i]/6. + k2[i]/3. + k3[i]/3. + k4[i]/6.);
  }

  // Advance
  upp(f, df, f, dt);
}

__global__
void integrate(R *f, R *t, R *dt, R *tnext) {
  __shared__ R temp[nthreads * ndim];
  __shared__ R ttemp;
  __shared__ R TNEXT;
  __shared__ R DT;

  ttemp=*t;
  TNEXT=*tnext;
  DT=*dt;
  int gindex = (threadIdx.x + blockIdx.x * blockDim.x) * ndim;
  int lindex = threadIdx.x * ndim;

  // Read input elements into shared memory
  for(int i=0; i<ndim; ++i) {
    temp[lindex+i] = f[gindex+i];
  }

  // Synchronize threads
  __syncthreads();
  
  
  
  while (ttemp < TNEXT) { 
  // Perform the RK4 integration
  rk4(&temp[lindex], &ttemp, &DT);
  ttemp=ttemp+DT;
  }
  // Write the data back
  for(int i=0; i<ndim; ++i) {
    f[gindex+i] = temp[lindex+i];
  }
  // Synchronize threads
  __syncthreads();
}


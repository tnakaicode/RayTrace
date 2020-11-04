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

__host__
void init_values(float *y, int nx, int ndim) {
  for(int i=0;i<nx;i++) {
    y[i*ndim]= sin(2.*pi*(float)i/(float)(nx-1));
    y[i*ndim+1]= 0.;
  }
}
__device__
void eval_rhs(float *f, float *df) {
  assert ( ndim == 2 );
  df[0] = f[1];
  df[1] = -omegasq * f[0];
}

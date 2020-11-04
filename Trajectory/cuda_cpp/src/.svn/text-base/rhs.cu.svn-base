#include <cstdio>
#include <cassert>
#include <math.h>
#include "RHS/harmonic.h"
//
using namespace std;

/* For every problem (e.g., harmonic oscillator, particle_in_tmag ) 
create two files, one header file 'harmonic.h', and a corresponding
.cu file e.g, harmonic.cu. The main part of the code should be
gode.cu and the ode solver should be evolve.cu which would contain
the function rnkt4 (or rnkt2 or any other) */

__device__
void eval_rhs(float *f, float *df) {
  assert ( ndim == 2 );
  df[0] = f[1];
  df[1] = -omegasq * f[0];
}

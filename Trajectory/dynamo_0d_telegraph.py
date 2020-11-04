import numpy as np
import noise as noise
from pylab import *

# ---------------------
variable = 2
Nensemble = 100
istride = variable
N = variable * Nensemble
f = np.zeros(N, complex)
imag = complex(0, 1)
alpha0 = 1.
etat = 1.
Shear = 10.
k = 1.
tc = 0.1
alpha = alpha0 * np.ones(Nensemble)
tcounter = np.zeros(Nensemble)
# ---------------------


def model(y, t, dt, istep):
    global alpha, tcounter
    for iensemble in range(0, Nensemble):
        Bx = y[iensemble * istride + 0]
        By = y[iensemble * istride + 1]
        f[iensemble * istride + 0] = imag * k * \
            alpha[iensemble] * By - etat * k * k * Bx
    #  f[1]=-Shear*y[0]+imag*k*alpha*y[0]-etat*k*k*y[1];
        f[iensemble * istride + 1] = -Shear * y[0] - etat * k * k * By
        if istep == 1:
            alpha[iensemble], tcounter[iensemble] = noise.telegraph(
                1 / tc, 1 / tc, alpha0, -alpha0, t, alpha[iensemble], tcounter[iensemble])
#      print t,alpha[iensemble],tcounter[iensemble],iensemble,real(Bx*Bx.conjugate());
    return f


def dimension():
    print('Solving 0 d dynamo equations with telegraph noise')
    return N


def diagnostic(y, t, counter):
    menergy = 0
    for iensemble in range(0, Nensemble):
        Bx = y[iensemble * istride + 0]
        By = y[iensemble * istride + 1]
        energy = real(Bx * Bx.conjugate() + By * By.conjugate())
        menergy = menergy + energy
        if iensemble == 1:
            energy1 = energy
        if iensemble == 2:
            energy2 = energy
        if iensemble == 3:
            energy3 = energy
        if iensemble == 4:
            energy4 = energy
        if iensemble == 5:
            energy5 = energy
    menergy = menergy / Nensemble
    print(t, menergy, energy1, energy2, energy3, energy4, energy5)


def iniconf():
    amp = 1e-8
    for iensemble in range(0, Nensemble):
        rantheta = noise.uniform(-np.pi, np.pi)
        Bx = amp * complex(cos(rantheta), sin(rantheta))
        rantheta = noise.uniform(-np.pi, np.pi)
        By = amp * complex(cos(rantheta), sin(rantheta))
        f[iensemble * istride + 0] = Bx
        f[iensemble * istride + 1] = By
    return f

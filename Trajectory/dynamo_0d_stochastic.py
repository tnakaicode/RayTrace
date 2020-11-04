import numpy as np
import noise as noise
from pylab import *

variable = 2
Nensemble = 100
N = variable * Nensemble
istride = variable
f = np.zeros(N, complex)
alpha = np.zeros(N)
imag = complex(0, 1)
zzero = complex(0, 0)
alpha0 = 1.
etat = 1.
Shear = 10.
k = 1.
alpha_mean = 0
alpha_var = 0.1


def Smodel(y, t, dt, istep):
    alpha = noise.gaussian(alpha_mean, alpha_var, Nensemble)
    for iensemble in range(0, Nensemble):
        By = y[iensemble * istride + 1]
        alpha = alpha0 * alpha[iensemble]
        f[iensemble * istride + 0] = imag * k * alpha * By
        # f[1]=-Shear*y[0]+imag*k*alpha*y[0]-etat*k*k*y[1];
        f[iensemble * istride + 1] = 0.
        return f


def model(y, t, dt, istep):
    for iensemble in range(0, Nensemble):
        Bx = y[iensemble * istride + 0]
        By = y[iensemble * istride + 1]
        f[iensemble * istride + 0] = -etat * k * k * Bx
    #  f[1]=-Shear*y[0]+imag*k*alpha*y[0]-etat*k*k*y[1];
        f[iensemble * istride + 1] = -Shear * y[0] - etat * k * k * By
    return f


def dimension():
    print("Solving 0 D dynamo equations with white-in-time alpha:")
    return N


def diagnostic(y, t, counter):
    menergy = 0
    energy1 = 0
    energy2 = 0
    energy3 = 0
    energy4 = 0
    energy5 = 0
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
        Bx = amp * complex(np.cos(rantheta), np.sin(rantheta))
        rantheta = noise.uniform(-np.pi, np.pi)
        By = amp * complex(np.cos(rantheta), np.sin(rantheta))
        f[iensemble * istride + 0] = Bx
        f[iensemble * istride + 1] = By
    return f

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import time

sys.path.append(os.path.join("../"))
from base import plot2d
import PlotTrajectory.noise as noise

# ---------------------
variable = 2
Nensemble = 1024
N = variable * Nensemble
istride = variable
f = np.zeros(N)
etat = 1.
Shear = 1.
tc = 0.0001
alpha0 = np.sqrt(1. / tc)
nu = 1. / tc
tcounter = np.zeros(Nensemble)
alpha = alpha0 * np.ones(Nensemble)
# ---------------------


def model(y, t, dt, istep):
    global alpha, tcounter
    for iensemble in range(0, Nensemble):
        Bx = y[iensemble * istride + 0]
        By = y[iensemble * istride + 1]
        f[iensemble * istride + 0] = alpha[iensemble] * By - etat * Bx
        f[iensemble * istride + 1] = -Shear * Bx - etat * By
        if istep == 1:
            alpha[iensemble], tcounter[iensemble] = noise.telegraph(
                1 / tc, 1 / tc, alpha0, -alpha0, t, alpha[iensemble], tcounter[iensemble])
        #print (t,alpha[iensemble],tcounter[iensemble],iensemble,real(Bx*Bx.conjugate()))
    return f


def dimension():
    print("Solving dynamo model of Vishniac and Brandenburg (telegraph alpha):")
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
        energy = np.real(Bx * Bx + By * By)
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
        ranx = noise.uniform(0, 1.)
        Bx = amp * ranx
        rantheta = noise.uniform(0, 1.)
        By = amp * ranx
        f[iensemble * istride + 0] = Bx
        f[iensemble * istride + 1] = By
    return f

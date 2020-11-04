from pylab import *
import numpy as np
import noise

N = 1
f = np.zeros(N)
omega = 1
famp = 0.01
noise_mean = 0
noise_var = 1


def model(y, t, dt, istep):
    return 0


def Smodel(y, t, dt, istep):
    f[0] = famp * noise.gaussian(noise_mean, noise_var, 1)
    return f


def dimension():
    print("Langveing Equation of one variable")
    return N


def diagnostic(y, t, counter):
    energy = y[0] * y[0]
    print(t, y[0], energy)


def iniconf():
    f[0] = 0
    return f

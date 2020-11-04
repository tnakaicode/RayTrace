from pylab import *
import numpy as np
import noise

N = 2
f = np.zeros(N)
omega = 1
famp = 0.1
noise_mean = 0
noise_var = 0.1


def model(y, t, dt, istep):
    f[0] = y[1]
    f[1] = -omega * omega * y[0]
    return f


def Smodel(y, t, dt, istep):
    f[0] = 0
    f[1] = famp * noise.gaussian(noise_mean, noise_var, 1)
    return f


def dimension():
    print("Solving Harmonic Oscillator with stochastic forcing")
    return N


def diagnostic(y, t, counter):
    energy = y[1] * y[1] + omega * omega * y[0] * y[0]
    print(t, y[0], y[1], energy)


def iniconf():
    y0 = [0, 1]
    return y0

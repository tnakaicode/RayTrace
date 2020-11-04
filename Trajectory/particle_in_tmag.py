import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import time
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join("../"))
from base import plot2d
import PlotTrajectory.noise as noise
import PlotTrajectory.technical as technical

# ---------------------
# Bfield='ABC_c';
Bfield = 'ABC'
A = 1.
B = 1.
C = 1.
Nparticle = 40960
# m=1.5
kk = 1. / 10.
qbym = 1.
omega = 1. / 10.
dim = 3
l_lyapunov = 0
mag_BB = 1.
mag_BBc = 0.
if l_lyapunov == 1:
    ddim = 4 * dim
    snumber = 0.1
else:
    ddim = 2 * dim
pstride = ddim
N = Nparticle * (ddim)
f = np.zeros(N)
y0 = np.zeros(N)
# ---------------------


def model(y, t, dt, istep):
    # y[0:2] is position, y[3:5] is momentum
    for ip in range(0, Nparticle):
        x = y[ip * pstride + 0 * dim:ip * pstride + 1 * dim]
        v = y[ip * pstride + 1 * dim:ip * pstride + 2 * dim]
        if l_lyapunov == 1:
            dx = y[ip * pstride + 2 * dim:ip * pstride + 3 * dim]
            dv = y[ip * pstride + 3 * dim:ip * pstride + 4 * dim]
            dBBdx = magfield(x, 1)
        BB = magfield(x, 0)
        EE = BB * np.cos(omega * t) * (omega / kk)
        BB = BB * np.sin(omega * t)
        f[ip * pstride + 0 * dim:ip * pstride + 1 * dim] = v
        f[ip * pstride + 1 * dim:ip * pstride +
            2 * dim] = qbym * (EE + np.cross(v, BB))
        # if l_lyapunov==1:
        #f[ip*pstride+2*dim:ip*pstride+3*dim] = dv;
        #          f[ip*pstride+3*dim:ip*pstride+4*dim] = qbym * ( np.cross(dv,BB)+np.cross(v-dv,np.dot(dBBdx,dx)) )
    # print f
    return f


def dimension():
    return N


def modsqr(A):
    size = len(A)
    xx = 0
    for ix in range(0, size):
        xx = xx + A[ix] * A[ix]
    return xx


def magfield(x, der):
    # Create array for the magnetic field
    if Bfield == 'default':
        print('no default magnetic field set')
        technical.end_program()
    elif Bfield == 'mirror':
        magfield, dmagfield = mirror(x)
    elif Bfield == 'ABC':
        magfield, dmagfield = ABC(x)
    elif Bfield == 'ABC_c':
        magfield, dmagfield = ABC_c(x)
    else:
        print('No such magnetic field')
        technical.end_program()
    if der == 1:
        return dmagfield
    else:
        return magfield


def mirror(x):
    BB = np.zeros(dim)
    dBB = np.zeros([dim, dim])
    BB[2] = (m * x[2]) + 1
    BB[1] = -(0.5) * (m * x[0]) / (x[0] * x[0] + x[1] * x[1])
    BB[0] = -(0.5) * (m * x[1]) / (x[0] * x[0] + x[1] * x[1])
    return BB, dBB


def ABC(x):
    BB = np.zeros(dim)
    dBB = np.zeros([dim, dim])
    xx = x[0]
    yy = x[1]
    zz = x[2]
    BB[0] = A * np.cos(kk * yy) + C * np.sin(kk * zz)
    dBB[0, 0] = 0
    dBB[0, 1] = -A * np.sin(kk * yy)
    dBB[0, 2] = C * np.cos(kk * zz)
    BB[1] = B * np.sin(kk * xx) + A * np.cos(kk * zz)
    dBB[1, 0] = B * np.cos(kk * xx)
    dBB[1, 1] = 0
    dBB[1, 2] = -A * np.sin(kk * zz)
    BB[2] = C * np.sin(kk * yy) + B * np.cos(kk * xx)
    dBB[2, 0] = -B * np.sin(kk * xx)
    dBB[2, 1] = C * np.cos(kk * yy)
    dBB[2, 2] = 0
    return BB, dBB * kk


def ABC_c(x):
    # Easiest to just call ABC to avoid code repetition, and dBB isn't even affected by Bz
    # BB=np.zeros(dim);
    # dBB=np.zeros([dim,dim]);
    BB, dBB = ABC(x)
    Bc = np.zeros(dim)  # Initialize our Bc constant magnetic field
    Bc[2] = mag_BBc * 1  # Set Bz!
    # Temp
    # print BB, dBB
    # print "now constantized"
    BB = mag_BB * BB + Bc     # Add our constant magnetic field, change magnitude
    dBB = mag_BB * dBB        # Derivate responds to magnitude
    # print BB, dBB
    return BB, dBB


def iniconf(lparam2file=False):
    print('setting initial conditions')
    for ip in range(0, Nparticle):
        y0[ip * pstride + 0] = 2 * 3.14 * np.random.ranf()
        y0[ip * pstride + 1] = 0
        y0[ip * pstride + 2] = 0
        y0[ip * pstride + 3] = 0.01  # np.random.ranf()
        y0[ip * pstride + 4] = 0
        y0[ip * pstride + 5] = 0.
        if l_lyapunov == 1:
            y0[ip * pstride + 6] = snumber * (1. / kk) * noise.uniform(-1, 1)
    return y0


def diagnostic(y, t, counter, fname, ldiag2file):
    energy = 0.
    rsqr = 0
    mFTLE = 0
    for ip in range(0, Nparticle):
        x = y[ip * pstride + 0:ip * pstride + dim]
        x0 = y0[ip * pstride + 0:ip * pstride + dim]
        rsqr = rsqr + modsqr(x - x0)
        v = y[ip * pstride + dim:ip * pstride + (2 * dim)]
        energy = energy + modsqr(v)
        if l_lyapunov == 1:
            dx = y[ip * pstride + (2 * dim):ip * pstride + (3 * dim)]
            dxsqrt = np.sqrt(modsqr(dx))
            dx0 = y0[ip * pstride + (2 * dim):ip * pstride + (3 * dim)]
            dx0sqrt = np.sqrt(modsqr(dx0))
            if t == 0:
                FTLE = 0
            else:
                FTLE = (1. / t) * np.log(dxsqrt) / np.log(dx0sqrt)
            mFTLE = mFTLE + FTLE
    # print t,energy/Nparticle,rsqr/Nparticle,mFTLE/Nparticle;
    energy = energy / Nparticle
    if ldiag2file == 1:
        fname.write("{}\t".format(t))
        fname.write("{:.2f}\t".format(x[0]))
        fname.write("{:.2f}\t".format(x[1]))
        fname.write("{:.2f}\t".format(x[2]))
        fname.write("{:.2f}\n".format(energy))
        print("current tstep: {}".format(t))
    else:
        print(t, x[0], x[1], x[2], energy)

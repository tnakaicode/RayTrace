import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import math
import time
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.join("../"))
from base import plot2d
import Trajectory.noise as noise
import Trajectory.technical as technical


Bfield = "dipole"
Nparticle = 1
m = 200
kk = 32.
qbym = 1
dim = 3
l_lyapunov = 0
mag_BB = 0.524
mag_BBc = 0.
angle = 0
if l_lyapunov == 1:
    ddim = 4 * dim
    snumber = 0.1
else:
    ddim = 2 * dim

pstride = ddim
N = Nparticle * (ddim)
f = np.zeros(N)
y0 = np.zeros(N)


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
        f[ip * pstride + 0 * dim:ip * pstride + 1 * dim] = v
        f[ip * pstride + 1 * dim:ip * pstride +
            2 * dim] = qbym * np.cross(v, BB)
        if l_lyapunov == 1:
            f[ip * pstride + 2 * dim:ip * pstride + 3 * dim] = dv
            f[ip * pstride + 3 * dim:ip * pstride + 4 * dim] = qbym * \
                (np.cross(dv, BB) + np.cross(v - dv, np.dot(dBBdx, dx)))
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
    if Bfield == "default":
        print("no default magnetic field set")
        technical.end_program()
    elif Bfield == "dipole":
        magfield, dmagfield = dipole(x)
    else:
        print("No such magnetic field")
        technical.end_program()
    if der == 1:
        return dmagfield
    else:
        return magfield


def dipole(x):
    BBx = np.zeros(dim)
    BB = np.zeros(dim)
    dBB = np.zeros([dim, dim])
    BBx[1] = mag_BB * np.sin(angle)
    BBx[2] = mag_BB * np.cos(angle)
    r = np.sqrt(modsqr(x))
    BB[0] = (3 * x[0] * m * x[2]) / (r**5)
    BB[1] = (3 * x[1] * m * x[2]) / (r**5)
    BB[2] = ((3 * (x[2]**2) * m) / (r**5)) - (m / (r**3))
    BB = BB + BBx
    12
    return BB, dBB


def iniconf():
    for ip in range(0, Nparticle):
        y0[ip * pstride + 0] = 2 * 3.141592  # *np.random.ranf()
        y0[ip * pstride + 1] = 0
        y0[ip * pstride + 2] = 0
        y0[ip * pstride + 3] = 0.1  # np.random.ranf()
        y0[ip * pstride + 4] = 0
        y0[ip * pstride + 5] = 0.1
        if l_lyapunov == 1:
            y0[ip * pstride + 6] = snumber * (1. / kk) * noise.uniform(-1, 1)
    return y0


def diagnostic(y, t, counter, fname, ldiag2file):
    print()


#energy = 0
#rsqr = 0
#mFTLE = 0
# for ip in range(0, Nparticle):
#    x = y[ip * pstride + 0:ip * pstride + dim]
#    x0 = y0[ip * pstride + 0:ip * pstride + dim]
#    rsqr = rsqr + modsqr(x - x0)
#    v = y[ip * pstride + dim:ip * pstride + (2 * dim)]
#    energy = energy + modsqr(v)
#    if l_lyapunov == 1:
#        dx = y[ip * pstride + (2 * dim):ip * pstride + (3 * dim)]
#    dxsqrt = np.sqrt(modsqr(dx))
#    dx0 = y0[ip * pstride + (2 * dim):ip * pstride + (3 * dim)]
#    dx0sqrt = np.sqrt(modsqr(dx0))
#
# if t == 0:
#FTLE = 0
# else:
#FTLE = (1. / t) * np.log(dxsqrt / dx0sqrt)
#mFTLE = mFTLE + FTLE
# print t,energy/Nparticle,rsqr/Nparticle,mFTLE/Nparticle;
# if ldiag2file == 1:
# fname.write(str(t) + "\t" + str(x[0]) + "\t" + str(x[1]) +
#            "\t" + str(x[2]) + "\t" + str(rsqr) + "\t" + str(mFTLE) + "\n")
#vel = open("vel.txt", "a")
#vel.write(str(v[0]) + " " + str(v[1]) + " " + str(v[2]) + "\n")
# vel.close()
# print "current tstep: " + str(t)
# else:
# 13
# print t, x[0], x[1], x[2], rsqr, mFTLE

# A.2 Other Files
# For the access to the rest of the code, see http: // code.google.com / p / pyoden / source/
# browse/  # svn%2Ftrunk. It might be particularly useful to look at odeN.py, where the step
# time and the final time i
#

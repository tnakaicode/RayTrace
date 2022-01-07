import numpy as np
import noise
import technical
import misc
from pylab import *
from mpl_toolkits.mplot3d import Axes3D

Ufield = 'Roberts_xy'
l_lyapunov = 0
lpassive_angle = 0
# mu is the dynamic viscosity
mu = 1.
Nparticle = 1
# m=1.5
kk = 4.
mag_UU = 0.01
# the two principal axis of the ellipsoid
# We assume that radA is the semi-major axis > radB
# their ratio is lambda. Here we give only radA and lambda-1
# as input, and deduce the rest. This lambda is equal to beta in the paper
# Zhang el al internationl journal of multiphase flows, 2001
radA = 1.
lambda_minus_1 = 0.2
rad_lambda = 1. + lambda_minus_1
radB = radA / rad_lambda
lambda2_minus_1 = lambda_minus_1 * (rad_lambda + 1)
sqrt_lambda2_minus_1 = np.sqrt(lambda2_minus_1)
alpha0 = (rad_lambda**2 / lambda2_minus_1) + (1. / 2.) * log((rad_lambda - sqrt_lambda2_minus_1) /
                                                             (rad_lambda - sqrt_lambda2_minus_1)) * (rad_lambda / (lambda2_minus_1)**(3. / 2.))
# the next expression is not the same in Zhang et al and Mortensen et al. There is an overall problem with sign. I do not see how
# gamma0 can be negative taken from Jeffery's paper, hence we have taken the expression of Mortensen et al.
gamma0 = (2. / lambda2_minus_1) + np.log((rad_lambda - sqrt_lambda2_minus_1) /
                                         (rad_lambda - sqrt_lambda2_minus_1)) * (rad_lambda / (lambda2_minus_1)**(3. / 2.))
pdenom = alpha0 + gamma0 * rad_lambda**2
prefactor = (16. / 3.) * np.pi * mu * rad_lambda * radA**3 / pdenom
# ---- the I_zz for such an ellisoid ------
# mass of the particle
rhop = 1.
mp = rhop * pi * radA * radB
Izz = (mp / 5.) * (1 + rad_lambda**2) * radA**2
tauS1 = 6 * pi * mu * radA / mp
tauS = 1. / tauS1
# the two components of the resistence tenson in diagonal form
# (these expression may need to be rechecked ) The are not the same expression
# in Zhang et al and Mortensen et al Phys. of Fluids 2008. The discrepency is the sign of the second term in the
# denominator of k11, For us k11 = kzz of those papers. We have used the expression in Mortensen
k11 = 8. * (lambda2_minus_1**(3. / 2.)) / ((rad_lambda**2 + lambda2_minus_1)
                                           * log(rad_lambda - sqrt_lambda2_minus_1) + rad_lambda * sqrt_lambda2_minus_1)
k22 = 16. * (lambda2_minus_1**(3. / 2.)) / ((2. * lambda2_minus_1 - 1) *
                                            log(rad_lambda - sqrt_lambda2_minus_1) + rad_lambda * sqrt_lambda2_minus_1)
#
dim = 2
Kres_diag = np.zeros([dim, dim])
Kres_diag[0, 0] = k11
Kres_diag[1, 1] = k22
Kres = np.zeros([dim, dim])
ddim = 2 * dim + 2
pstride = ddim
N = Nparticle * (ddim)
f = np.zeros(N)
y0 = np.zeros(N)


def model(y, t, dt, istep):
    # y[0:3] is position and angle, y[3:5] is momentum and angular momentum
    for ip in range(0, Nparticle):
        x = y[ip * pstride + 0 * dim:ip * pstride + 1 * dim]
        v = y[ip * pstride + 1 * dim:ip * pstride + 2 * dim]
        theta = misc.periodic(
            y[ip * pstride + 2 * dim:ip * pstride + 2 * dim + 1], 2 * pi)
        y[ip * pstride + 2 * dim:ip * pstride + 2 * dim + 1] = theta
        ell = y[ip * pstride + 2 * dim + 1:ip * pstride + 2 * dim + 2]
# the rotation matrix maps the unit vector along the x direction
# to the direction of the semi-major axis.
        Kres = rot_matrix2d(Kres_diag, -theta)
# The above line may be coded more efficiently because Kres_diag is diagonal
# hence the result can be found with less multiplications.
        omega_p = ell / Izz
        UU, curlu, Sij = vel(x, 1)
        curlu_body = curlu
        Sij_body = rot_matrix2d(Sij, theta)
        JefH = Sij_body[0, 1]
        Jef_zeta = curlu_body / 2.
# the prefactor may need revision in the list of the fact that now we have reduced the
# the problem to two dimensions
        Jeffery_N = prefactor * (lambda_minus_1 * (rad_lambda + 1)
                                 * JefH + (1 + rad_lambda**2) * (Jef_zeta - omega_p))
        f[ip * pstride + 0 * dim:ip * pstride + 1 * dim] = v
        if lpassive_angle == 1:
            f[ip * pstride + 1 * dim:ip * pstride +
                2 * dim] = (tauS1) * (UU - v)
        else:
            f[ip * pstride + 1 * dim:ip * pstride + 2 *
                dim] = (tauS1) * np.dot(Kres, UU - v)
        f[ip * pstride + 2 * dim:ip * pstride + 2 * dim + 1] = omega_p
        f[ip * pstride + 2 * dim + 1:ip * pstride + 2 * dim + 2] = Jeffery_N
    # print f
    return f

#************************************#


def rot_matrix2d(AA, theta):
    Rot = np.zeros([2, 2])
    Rot[0, 0] = cos(theta)
    Rot[0, 1] = -sin(theta)
    Rot[1, 0] = sin(theta)
    Rot[1, 1] = cos(theta)
    Arot = np.dot(AA, Rot)
    Arot = np.dot(np.transpose(Rot), Arot)
    return Arot


def dimension():
    return N


def modsqr(A):
    size = len(A)
    xx = 0
    for ix in range(0, size):
        xx = xx + A[ix] * A[ix]
    return xx


def vel(x, der):
    # Create array for the velocity field
    if Ufield == 'default':
        print('no default velocity field set')
        technical.end_program()
    elif Ufield == 'constant':
        ufield = 1.
        strain = 0
        omega = 0.
    elif Ufield == 'Roberts_xy':
        ufield, omega, strain = Roberts_xy(x)
    elif Ufield == 'ABC':
        ufield, dufield = ABC(x)
    else:
        print('No such magnetic field')
        technical.end_program()
    if der == 0:
        return ufield
    else:
        return ufield, omega, strain


def Roberts_xy(x):
    UU = np.zeros(dim)
    dUU = np.zeros([dim, dim])
    Sij = np.zeros([dim, dim])
    xx = x[0]
    yy = x[1]
    UU[0] = -cos(kk * xx) * sin(kk * yy)
    dUU[0, 0] = sin(kk * xx) * sin(kk * yy)
    dUU[0, 1] = -cos(kk * xx) * cos(kk * yy)
    UU[1] = sin(kk * xx) * cos(kk * yy)
    dUU[1, 0] = cos(kk * xx) * cos(kk * yy)
    dUU[1, 1] = -sin(kk * xx) * sin(kk * yy)
    Sij = (0.5) * (dUU + np.transpose(dUU))
    omega = dUU[0, 1] - dUU[1, 0]
    return UU, omega * kk, Sij * kk


def ABC(x):
    BB = np.zeros(dim)
    dBB = np.zeros([dim, dim])
    xx = x[0]
    yy = x[1]
    zz = x[2]
    BB[0] = cos(kk * yy) + sin(kk * zz)
    dBB[0, 0] = 0
    dBB[0, 1] = -sin(kk * yy)
    dBB[0, 2] = cos(kk * zz)
    BB[1] = sin(kk * xx) + cos(kk * zz)
    dBB[1, 0] = cos(kk * xx)
    dBB[1, 1] = 0
    dBB[1, 2] = -sin(kk * zz)
    BB[2] = sin(kk * yy) + cos(kk * xx)
    dBB[2, 0] = -sin(kk * xx)
    dBB[2, 1] = cos(kk * yy)
    dBB[2, 2] = 0
    return BB, dBB * kk


def iniconf(lparam2file):
    for ip in range(0, Nparticle):
        #        y0[ip*pstride+0]=2*3.14*np.random.ranf();
        y0[ip * pstride + 0] = 1.
        y0[ip * pstride + 1] = 1.
        y0[ip * pstride + 2] = 0
        y0[ip * pstride + 3] = 0.  # np.random.ranf()
        y0[ip * pstride + 4] = 0
        y0[ip * pstride + 5] = 0.
        #       if l_lyapunov==1:
        #    y0[ip*pstride+6]=snumber*(1./kk)*noise.uniform(-1,1);
    tau_flu = 1. / (kk * mag_UU)
    St = tauS / tau_flu
    if lparam2file == 1:
        fname.write(str(St) + "\n")
    else:
        print('St=', St)
    return y0


def getW4plot(x0=0, y0=0, x1=2 * pi, y1=2 * pi, Ngrid=64):
    WW = np.zeros([Ngrid, Ngrid])
    xx = np.zeros(Ngrid)
    yy = np.zeros(Ngrid)
    for igrid in range(0, Ngrid):
        for jgrid in range(0, Ngrid):
            dx = (x1 - x0) / Ngrid
            dy = (y1 - y0) / Ngrid
            xx[igrid] = x0 + dx * igrid
            yy[jgrid] = y0 + dy * jgrid
            xvec = [x0 + dx * igrid, y0 + dy * jgrid]
            UU, curlu, Sij = vel(xvec, 1)
            WW[igrid, jgrid] = curlu
    return xx, yy, WW


def diagnostic(y, t, counter, fname, ldiag2file):
    energy = 0
    rsqr = 0
    mFTLE = 0
    for ip in range(0, Nparticle):
        x = y[ip * pstride + 0:ip * pstride + dim]
        x0 = y0[ip * pstride + 0:ip * pstride + dim]
        rsqr = rsqr + modsqr(x - x0)
        v = y[ip * pstride + dim:ip * pstride + (2 * dim)]
        energy = energy + modsqr(v)
        theta = y[ip * pstride + (2 * dim):ip * pstride + (2 * dim) + 1]
        # if l_lyapunov == 1 :
        #  dx=y[ip*pstride+(2*dim):ip*pstride+(3*dim)];
        #  dxsqrt= np.sqrt(modsqr(dx));
        #  dx0=y0[ip*pstride+(2*dim):ip*pstride+(3*dim)];
        #  dx0sqrt= np.sqrt(modsqr(dx0));
        #  if t == 0:
        #      FTLE=0
        #  else:
        #      FTLE=(1./t)*np.log(dxsqrt)/np.log(dx0sqrt)
        #  mFTLE=mFTLE+FTLE
    # print t,energy/Nparticle,rsqr/Nparticle,mFTLE/Nparticle;
    if ldiag2file == 1:
        fname.write(str(t) + "\t" +
                    str(x[0]) + "\t" + str(x[1]) + "\t" + str(theta[0]) + "\n")
        print("current tstep: " + str(t))
    else:
        print(t, x[0], x[1], theta)

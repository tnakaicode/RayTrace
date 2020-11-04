import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D


def read_snap(fname, model='particle_in_mag', lflag=True):
    ytt = np.loadtxt(fname)
    time = ytt[0]
    yt = ytt[1:]
    NN = len(yt)
    if lflag == True:
        pstride = 4 * 3
    else:
        pstride = 2 * 3
    Nparticle = NN / (pstride)
    xp = np.zeros(Nparticle)
    yp = np.zeros(Nparticle)
    zp = np.zeros(Nparticle)
    vxp = np.zeros(Nparticle)
    vyp = np.zeros(Nparticle)
    vzp = np.zeros(Nparticle)
    if lflag == True:
        dxp = np.zeros(Nparticle)
        dyp = np.zeros(Nparticle)
        dzp = np.zeros(Nparticle)
        dvxp = np.zeros(Nparticle)
        dvyp = np.zeros(Nparticle)
        dvzp = np.zeros(Nparticle)
    if (model == 'particle_in_mag') or (model == 'particle_in_tmag'):
        print('reading data')
        for ip in range(0, Nparticle):
            xp[ip] = yt[ip * pstride + 0]
            yp[ip] = yt[ip * pstride + 1]
            zp[ip] = yt[ip * pstride + 2]
            vxp[ip] = yt[ip * pstride + 3]
            vyp[ip] = yt[ip * pstride + 4]
            vzp[ip] = yt[ip * pstride + 5]
            if lflag == True:
                dxp[ip] = yt[ip * pstride + 6]
                dyp[ip] = yt[ip * pstride + 7]
                dzp[ip] = yt[ip * pstride + 8]
                dvxp[ip] = yt[ip * pstride + 9]
                dvyp[ip] = yt[ip * pstride + 10]
                dvzp[ip] = yt[ip * pstride + 11]
    if lflag == True:
        return time, xp, yp, zp, vxp, vyp, vzp, dxp, dyp, dzp, dvxp, dvyp, dvzp
    else:
        return time, xp, yp, zp, vxp, vyp, vzp

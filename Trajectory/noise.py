import numpy as np
import sys
import os
from pylab import *


def end_program():
    sys.exit(0)


def zero_matrix(row, column, datatype):
    pre_matrix = np.zeros((row, column), dtype=datatype)
    matrix = pre_matrix.view(type=np.matrix)
    return matrix


def telegraph_symmetric(nu, t, var, tcounter):
    tresident = t - tcounter
    ft = 0.5 - 0.5 * np.exp(-2 * nu * tresident)
    rr = np.random.ranf()
    if rr >= ft:
        xx = var
    else:
        xx = -var
        tcounter = t
    return xx, tcounter


def telegraph(nu1, nu2, a1, a2, t, var, tcounter):
    if (nu1 == nu2 and a1 == -a2):
        xx, tcounter = telegraph_symmetric(nu1, t, var, tcounter)
    else:
        print('not coded in the general case')
        end_program()
    return xx, tcounter


def gaussian(mean, sigma, nsample=2, datatype=np.float):
    grand = np.zeros(nsample, dtype=datatype)
    for i in range(0, nsample / 2):
        ran1 = np.random.ranf()
        ran2 = np.random.ranf()
        grand[2 * i] = np.sqrt(-2 * sigma * np.log(ran1)
                               ) * np.cos(2. * np.pi * ran2)
        grand[2 * i + 1] = np.sqrt(-2 * sigma *
                                   np.log(ran1)) * np.sin(2 * np.pi * ran2)
    if nsample % 2 == 1:
        ran1 = np.random.ranf()
        ran2 = np.random.ranf()
        grand[nsample - 1] = np.sqrt(-2 * sigma *
                                     np.log(ran1)) * np.cos(2. * np.pi * ran2)
    grand = grand + mean
    return grand


def uniform(min, max):
    xrange = max - min
    x = xrange * np.random.ranf() + min
    return x

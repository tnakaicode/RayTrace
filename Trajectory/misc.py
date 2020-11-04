#!/opt/local/bin/python2.7
# Filename: misc.py
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import platform
import subprocess
from mpl_toolkits.mplot3d import Axes3D


def periodic(psi, hlimit=2 * np.pi):
    # sets psi to be periodic between zero and hlimit
    psi_periodic = psi % hlimit
    return psi_periodic


def plot3d(xx, yy, zz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(xx, yy, zz)
    plt.plot([xx[0]], [yy[0]], [zz[0]], '*r')
    plt.xlabel('kx')
    plt.ylabel('ky')


def scatter3d(xx, yy, zz):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xx, yy, zz)

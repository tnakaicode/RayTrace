# Contains the time-stepping routines
# This module is divided into two parts
# (i) where the RHS of the ODE is calculated
# (ii) schemes of integration
# Dhruba : 15 July 2008
# -----------------%---------------
# mod timestep.py

import numpy as np
from pylab import *
import equ

# reload the moduel to make sure
# that new changes are incorporated
# reload(equ)
#
# ==================================
# Integration schemes start below:
# ==================================
#


def euler(y0, t, dt, counter):
    if counter == 1:
        print(" Integrating by Euler scheme")
    istep = 1
    f = equ.RHS(y0, t, dt, istep)
    y = y0 + f * dt
    return y, y0


def adbsh(y0, t, dt, counter):
    if counter == 1:
        print(" Integrating by Adams-Bashforth 2nd order scheme")
    istep = 1
    f = equ.RHS(y0, t, dt, istep)
    y = y0 + f * dt
    return y, y0


def rk2(y0, t, dt, counter):
    if counter == 1:
        print(" Integrating by Runge-Kutta 2nd order scheme")
    istep = 1
    k1 = dt * equ.RHS(y0, t, dt, istep)
    yhalf = y0 + k1 / 2.
    thalf = t + dt / 2.
    istep = 2
    k2 = dt * equ.RHS(yhalf, thalf, dt, istep)
    y = y0 + k2
    return y, y0


def rk4(y0, t, dt, counter):
    if counter == 1:
        print(" Integrating by Runge-Kutta 4th order scheme")
    istep = 1
    k1 = dt * equ.RHS(y0, t, dt, istep)
    yhalf = y0 + k1 / 2.
    thalf = t + dt / 2.
    istep = 2
    k2 = dt * equ.RHS(yhalf, thalf, dt, istep)
    yhalf = y0 + k2 / 2.
    thalf = t + dt / 2.
    istep = 3
    k3 = dt * equ.RHS(yhalf, thalf, dt, istep)
    yhalf = y0 + k3
    thalf = t + dt
    istep = 4
    k4 = dt * equ.RHS(yhalf, thalf, dt, istep)
    y = y0 + k1 / 6. + k2 / 3. + k3 / 3. + k4 / 6.
    return y, y0


def euler_marayuma(y, y0, t, dt, counter):
    if counter == 1:
        print("Integrating by euler-marayuma methods")
    istep = 1
    f = equ.SRHS(y0, t, dt, istep)
    y = y + f * np.sqrt(dt)
    return y

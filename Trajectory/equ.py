# Equation of the form
# dy/dt = f(y,t,\eta1) + \eta2
# where \eta1 and \eta2 are noise. This
# module contains the equation, initial
# condition etc.
# 15 July 2008 : Dhruba
# --------------------
# mod equ.py
from pylab import *
import numpy as np
import technical


def iniconf(lparam2file=False):
    global N
    global y, y0, f, eta1, eta2
    N = M.dimension()
    print('Model variables are real')
    y0 = np.zeros(N)
    y = np.zeros(N)
    f = np.zeros(N)
    if lparam2file:
        y0 = M.iniconf(lparam2file)
    else:
        y0 = M.iniconf()
        # y0[0]=1.
        # y0[3]=1.
    return None


def iniconf_complex():
    global N
    global y, y0, f, eta1, eta2
    N = M.dimension()
    print('Model variables are complex')
    y0 = zeros(N, complex)
    y = zeros(N, complex)
    f = zeros(N, complex)
    #  eta1=zeros(N,complex);
    # eta2=zeros(N,complex);
    y0 = M.iniconf()
    return None


def RHS(y, t, dt, istep):
    f = M.model(y, t, dt, istep)
    return f


def SRHS(y0, t, dt, istep):
    f = M.Smodel(y, t, dt, istep)
    return f


def diagnostic(y, t, counter, diagnostic_file, ldiag2file):
    M.diagnostic(y, t, counter, diagnostic_file, ldiag2file)


def select_model(modelname):
    global M
    deterministic = 0
    stochastic = 0
    if modelname == 'default':
        print('no default model set')
        technical.end_program()
    elif modelname == 'HarmonicOsc1d':
        import HarmonicOsc1d as M
        modtype = 'real'
        deterministic = 1
        stochastic = 0
    elif modelname == 'HarmonicOsc1dS':
        import HarmonicOsc1dS as M
        modtype = 'real'
        deterministic = 1
        stochastic = 1
    elif modelname == 'Langvein':
        import Langvein as M
        modtype = 'real'
        deterministic = 0
        stochastic = 1
    elif modelname == 'Dynamo2D':
        print('no such model')
        tehcnical.end_program()
    elif modelname == 'particle_in_mag':
        import particle_in_mag as M
        modtype = 'real'
        deterministic = 1
        stochastic = 0
    elif modelname == 'particle_in_tmag':
        import particle_in_tmag as M
        modtype = 'real'
        deterministic = 1
        stochastic = 0
    elif modelname == 'inertial_ellipsoid2d':
        import inertial_ellipsoid2d as M
        modtype = 'real'
        deterministic = 1
        stochastic = 0
    elif modelname == 'dynamo_0d_telegraph':
        import dynamo_0d_telegraph as M
        modtype = 'complex'
        deterministic = 1
        stochastic = 0
    elif modelname == 'dynamo_0d_stochastic':
        import dynamo_0d_stochastic as M
        modtype = 'complex'
        deterministic = 1
        stochastic = 1
    elif modelname == 'vishniac_brandenburg':
        import vishniac_brandenburg as M
        modtype = 'real'
        deterministic = 1
        stochastic = 1
    elif modelname == 'vb_telegraph':
        import vb_telegraph as M
        modtype = 'real'
        deterministic = 1
        stochastic = 0
    elif modelname == 'tracer_in_fluid':
        import tracer_in_fluid as M
        modtype = 'real'
        deterministic = 1
        stochastic = 0
    else:
        print(modelname, '\t', ':Model not found')
        technical.end_program()
    print('Solving for', modelname)
    return modtype, deterministic, stochastic

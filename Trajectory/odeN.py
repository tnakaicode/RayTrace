# Solve N first-order differential equation of the form:
#      dy/dt = f(y,t) + (stochastic noise)
# ------------
# 15 July 2008 : Dhruba
# ----------------------
""" Evolves in time N differential equation of the
  form :   dy/dt = f(y,t) + (stochastic noise)"""
# First import the needed modules
# also module for plotting
# odeN.py
import numpy as np
import timestep
import time
import sys
import equ
import os
import technical
from pylab import *
from matplotlib import *

model = 'none'
# -----------------
# Reload the modules to make sure
# that any new change has been incorporated
# ------ Input parameters ------------
# model='dynamo_0d_telegraph';
# model='vb_telegraph';
# model='dynamo_0d_stochastic';
# model='HarmonicOsc1d';
# model='HarmonicOsc1dS';
# model='Langvein';
model = 'particle_in_mag'
#model = 'particle_in_tmag'
# model='vishniac_brandenburg';
# model='polymer_in_fluid';
# model='tracer_in_fluid';
# model='inertial_ellipsoid2d';
dt = 1e-2
TMAX = 1.0

ldiag2file = 1
lparam2file = 0
#	Specify options for snapshotting
#	Current implementation of storing snapshots is storing them non-binary, in human-readable format
#	Along with the current time
l_storesnap = 0  # Boolean defining whether to snapshot or not

# Set this if you want a manual snap-stride size!
# dtstoresnap = 0.1	  #Float defining timesteps for storing snapshots
# Number of snapshots to store
wdtstoresnap = 0

snapcounter = 0
# ----------------


def main():
    global snapcounter, l_storesnap, wdtstoresnap, dtstoresnap, stridestoresnap
    modtype, deterministic, stochastic = equ.select_model(model)
    if modtype == 'complex':
        equ.iniconf_complex()
    elif modtype == 'real':
        equ.iniconf(lparam2file)
    else:
        print('modtype not recognised')
        print('modtype = ', modtype)
        technical.end_program()
    y0 = equ.y0
    y = equ.y
    f = equ.f
    # --------------
    #   Check args for timestep to start on;
    t = 0.0
    y = y0
    if len(sys.argv) > 1:
        startat = float(sys.argv[1])
        if int(startat) > TMAX or int(startat) == TMAX:
            sys.exit(
                'Failure in config, please specify larger TMAX or lower starting point!')
        y0 = np.loadtxt('snaps/snap_' + model + '_' + str(0.0))
        y0 = y0[1:]
        y = np.loadtxt('snaps/snap_' + model + '_' + str(startat))
        y = y[1:]
        t = startat
        # ---------------
    if l_storesnap == 1:
        dtstoresnap = (TMAX - t) / wdtstoresnap
        stridestoresnap = int(dtstoresnap / dt)

    kill = 0

    # Initial condition set
    print('Initial time: t = ' + repr(t))
    print('Initial guess of timestep: dt = ' + repr(dt))
    print('Integrating upto time TMAX = ' + repr(TMAX))
    # plot the initial condition
    # clf()
    #  plot([t],[0],'o')

    counter = 1
    if l_storesnap == 1:
        if os.path.exists('snaps'):
            print("Snapshot directory already exists, continuing")
        else:
            os.makedirs('snaps')
        if len(os.listdir('snaps')) > 0:
            confirm = 'x'
            while (confirm != 'n') and (confirm != 'y'):
                confirm = input(
                    'Snaps directory is not empty, continue? (y/n):')
            if confirm == 'n':
                sys.exit()
    if model == 'particle_in_mag':
        x1 = []
        x2 = []
        x3 = []

    # Integrate in time
    if (os.path.exists('diag_' + model)):
        print("Diagnostic file exists, renaming!")
        print('diag_' + model + time.strftime("%Y-%m-%d-%H-%M"))
        os.rename('diag_' + model, 'diag_' + model +
                  "_" + time.strftime("%Y-%m-%d-%H-%M"))
    diagnostic_file = open('diag_' + model, 'w')
    while (t < TMAX) and (kill != 1):
        equ.diagnostic(y, t, counter, diagnostic_file, ldiag2file)

        #	savestate:
        if l_storesnap == 1:
            if (dtstoresnap < dt):
                sys.exit('logical failure, dtstoresnap must be greater than dt')
            if snapcounter == 0 or t == TMAX or t == TMAX - dt:
                f = open('snaps/snap_' + model + '_' + str(t), 'wb')
                f.write(str(t) + "\n")
                for item in y:
                    if item != item:
                        print("ERROR; FOUND either nan or NaN!")
                        kill = 1
                        break
                    f.write(str(item) + "\n")
                f.close()
                snapcounter = stridestoresnap - 1
            else:
                snapcounter = snapcounter - 1

        if kill != 1:
            if model == 'particle_in_mag':
                x1.append(y[0])
                x2.append(y[1])
                x3.append(y[2])
            if deterministic == 1:
                y, y0 = timestep.rk4(y, t, dt, counter)
            if stochastic == 1:
                y = timestep.euler_marayuma(y, y0, t, dt, counter)
            t = t + dt

            counter = counter + 1
            #    plot([t],[(energy-energy0)/energy0],'bo')
            #    plot([t], y[0], 'bo')

    diagnostic_file.close()

    if model == 'particle_in_mag':
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(x1, x2, x3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        show()


if __name__ == '__main__':
    main()

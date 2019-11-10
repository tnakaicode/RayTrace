import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.constants as cnt
from optparse import OptionParser

from OCC.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.gp import gp_Ax1, gp_Ax2, gp_Ax3

from base import plotocc, Face, set_trf
from Surface import surf_curv


class TraceSystem (plotocc):

    def __init__(self):
        plotocc.__init__(self)
        self.axis = gp_Ax3(gp_Pnt(1000, 1000, 100), gp_Dir(0, 0, 1))
        self.trsf = set_trf(ax2=self.axis)

        ax1 = gp_Ax3(gp_Pnt(0, 0, 100), gp_Dir(0, 0, -1))
        ax1.Transform(self.trsf)
        self.surf1 = Face(ax1)
        self.surf1.rot_axs(rxyz=[0, 45, 10])

        ax2 = gp_Ax3(gp_Pnt(-500, 0, 100), gp_Dir(0, 0, 1))
        ax2.Transform(self.trsf)
        self.surf2 = Face(ax2)
        self.surf2.face = surf_curv(lxy=[200, 150], rxy=[0, 0])
        self.surf2.MoveSurface(ax2=self.surf2.axis)
        self.surf2.rot_axs(pxyz=[0,0,10], rxyz=[0, 45, 10])
        
        ax3 = gp_Ax3(gp_Pnt(-250, 0, 1000), gp_Dir(0, 0, 1))
        ax3.Transform(self.trsf)
        self.surf3 = Face(ax3)
        self.surf3.face = surf_curv(lxy=[1000, 1000], rxy=[0, 0])
        self.surf3.MoveSurface(ax2=self.surf3.axis)
        
        
    def Display(self):
        self.display.DisplayShape(self.surf1.face)
        self.display.DisplayShape(self.surf2.face)
        self.display.DisplayShape(self.surf3.face, transparency=0.7, color="BLUE")

        self.show_axs_pln(self.axis, scale=10)
        self.show_axs_pln(self.surf1.axis, scale=20)
        self.show_axs_pln(self.surf2.axis, scale=30)


if __name__ == "__main__":
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./")
    opt, argc = parser.parse_args(argvs)
    print(argc, opt)

    obj = TraceSystem()
    obj.Display()
    obj.show()

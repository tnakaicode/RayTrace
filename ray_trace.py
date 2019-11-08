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
        self.axis = gp_Ax3(gp_Pnt(0, 0, -100), gp_Dir(0, 0, 1))
        self.trsf = set_trf(ax2=self.axis)

        ax1 = gp_Ax3(gp_Pnt(0, 0, 100), gp_Dir(0, 1, 0))
        ax1.Transform(self.trsf)
        self.surf1 = Face(ax1)

        ax2 = gp_Ax3(gp_Pnt(0, 0, 200), gp_Dir(0, -1, 0))
        ax2.Transform(self.trsf)
        self.surf2 = Face(ax2)
        self.surf2.face = surf_curv(lxy=[200, 150], rxy=[0, 0])
        self.surf2.MoveSurface(self.surf2.axis)

        ax3 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        ax3.Transform(self.trsf)
        self.surf2.MoveSurface(ax3)

    def Display(self):
        self.display.DisplayShape(self.surf1.face)
        self.display.DisplayShape(self.surf2.face)

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

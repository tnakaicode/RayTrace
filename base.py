import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import pickle
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_XYZ
from OCC.Core.gp import gp_Lin
from OCC.Core.gp import gp_Mat, gp_GTrsf, gp_Trsf
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeWire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCCUtils.Construct import make_box, make_line, make_wire
from OCCUtils.Construct import make_plane, make_polygon
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import dir_to_vec, vec_to_dir

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

from Surface import surf_curv
from OCCQt import Viewer


def pnt_trf_vec(pnt=gp_Pnt(), vec=gp_Vec()):
    v = point_to_vector(pnt)
    v.Add(vec)
    return vector_to_point(v)


def set_trf(ax1=gp_Ax3(), ax2=gp_Ax3()):
    trf = gp_Trsf()
    trf.SetTransformation(ax2, ax1)
    return trf


def set_loc(ax1=gp_Ax3(), ax2=gp_Ax3()):
    trf = set_trf(ax1, ax2)
    loc = TopLoc_Location(trf)
    return loc


def trsf_scale(axs=gp_Ax3(), scale=1):
    trf = gp_Trsf()
    trf.SetDisplacement(gp_Ax3(), axs)
    return trf


def gen_ellipsoid(axs=gp_Ax3(), rxyz=[10, 20, 30]):
    sphere = BRepPrimAPI_MakeSphere(gp_Ax2(), 1).Solid()
    loc = set_loc(gp_Ax3(), axs)
    mat = gp_Mat(
        rxyz[0], 0, 0,
        0, rxyz[1], 0,
        0, 0, rxyz[2]
    )
    gtrf = gp_GTrsf(mat, gp_XYZ(0, 0, 0))
    ellips = BRepBuilderAPI_GTransform(sphere, gtrf).Shape()
    ellips.Location(loc)
    return ellips

def rot_axs(axis=gp_Ax3(), pxyz=[0, 0, 0], rxyz=[0, 0, 0]):
    axs = gp_Ax3(gp_Pnt(*pxyz), gp_Dir(0,0,1))
    ax1 = gp_Ax1(axis.Location(), axis.XDirection())
    ax2 = gp_Ax1(axis.Location(), axis.YDirection())
    ax3 = gp_Ax1(axis.Location(), axis.Direction())
    axs.Rotate(ax1, np.deg2rad(rxyz[0]))
    axs.Rotate(ax2, np.deg2rad(rxyz[1]))
    axs.Rotate(ax3, np.deg2rad(rxyz[2]))
    trsf = set_trf(gp_Ax3(), axs)
    axis.Transform(trsf)

class plotocc (Viewer):

    def __init__(self):
        self.display, self.start_display, self.add_menu, self.add_functionto_menu = init_display()
        Viewer.__init__(self)
        self.on_select()

    def show_box(self):
        self.display.DisplayShape(gp_Pnt())
        self.display.DisplayShape(make_box(100, 100, 100))

    def show_pnt(self, xyz=[0, 0, 0]):
        self.display.DisplayShape(gp_Pnt(*xyz))

    def show_ball(self, scale=100, trans=0.5):
        shape = BRepPrimAPI_MakeSphere(scale).Shape()
        self.display.DisplayShape(shape, transparency=trans)

    def show_ellipsoid(self, axs=gp_Ax3(), rxyz=[10., 10., 10.], trans=0.5):
        shape = gen_ellipsoid(axs, rxyz)
        self.display.DisplayShape(shape, transparency=trans, color="BLUE")
        return shape

    def show_axs_pln(self, axs=gp_Ax3(), scale=100):
        pnt = axs.Location()
        dx = axs.XDirection()
        dy = axs.YDirection()
        dz = axs.Direction()
        vx = dir_to_vec(dx).Scaled(1 * scale)
        vy = dir_to_vec(dy).Scaled(2 * scale)
        vz = dir_to_vec(dz).Scaled(3 * scale)

        pnt_x = pnt_trf_vec(pnt, vx)
        pnt_y = pnt_trf_vec(pnt, vy)
        pnt_z = pnt_trf_vec(pnt, vz)
        self.display.DisplayShape(pnt)
        self.display.DisplayShape(make_line(pnt, pnt_x), color="RED")
        self.display.DisplayShape(make_line(pnt, pnt_y), color="GREEN")
        self.display.DisplayShape(make_line(pnt, pnt_z), color="BLUE")

    def show_plane(self, axs=gp_Ax3(), scale=100):
        pnt = axs.Location()
        vec = dir_to_vec(axs.Direction())
        pln = make_plane(pnt, vec, -scale, scale, -scale, scale)
        self.display.DisplayShape(pln)

    def show(self):
        self.display.FitAll()
        self.start_display()


class Face (object):

    def __init__(self, axs=gp_Ax3(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axs
        self.face = surf_curv(rxy=[0, 0])
        self.MoveSurface(ax2=self.axis)

    def MoveSurface(self, ax1=gp_Ax3(), ax2=gp_Ax3()):
        trsf = set_trf(ax1, ax2)
        self.face.Move(TopLoc_Location(trsf))

    def rot_axs(self, pxyz=[0, 0, 0], rxyz=[0, 0, 0]):
        axs = gp_Ax3(gp_Pnt(*pxyz), gp_Dir(0,0,1))
        ax1 = gp_Ax1(self.axis.Location(), self.axis.XDirection())
        ax2 = gp_Ax1(self.axis.Location(), self.axis.YDirection())
        ax3 = gp_Ax1(self.axis.Location(), self.axis.Direction())
        axs.Rotate(ax1, np.deg2rad(rxyz[0]))
        axs.Rotate(ax2, np.deg2rad(rxyz[1]))
        axs.Rotate(ax3, np.deg2rad(rxyz[2]))
        trsf = set_trf(gp_Ax3(), axs)
        self.axis.Transform(trsf)
        self.face.Move(TopLoc_Location(trsf))
        
        

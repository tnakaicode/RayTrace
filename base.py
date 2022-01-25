import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import scipy as sp
import sys
import pickle
import json
import time
import os
import glob
import shutil
import datetime
import platform
import subprocess
import scipy.constants as cnt
from scipy.integrate import simps
from scipy import ndimage
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize, minimize_scalar, OptimizeResult
from optparse import OptionParser
from linecache import getline, clearcache
from matplotlib import animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('parso').setLevel(logging.ERROR)

from PyQt5.QtWidgets import QApplication, qApp
from PyQt5.QtWidgets import QDialog, QCheckBox
from PyQt5.QtWidgets import QFileDialog
# pip install PyQt5
# pip install --upgrade --force-reinstall PyQt5

from OCC.Display.SimpleGui import init_display
from OCC.Core.gp import gp_Cylinder, gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_XYZ
from OCC.Core.gp import gp_Lin, gp_Elips, gp_Pln, gp_Circ
from OCC.Core.gp import gp_Mat, gp_GTrsf, gp_Trsf
from OCC.Core.Geom import Geom_Curve, Geom_Plane, Geom_Line
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface, GeomAPI_ProjectPointOnSurf, GeomAPI_ProjectPointOnCurve
from OCC.Core.GeomAPI import GeomAPI_IntCS, GeomAPI_IntSS
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool, GeomLProp_CurveTool
from OCC.Core.GeomAbs import GeomAbs_C3, GeomAbs_G2
from OCC.Core.TopoDS import TopoDS_Shape, TopoDS_Compound, TopoDS_Face, topods_Face, topods_Solid
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt
from OCC.Core.TColgp import TColgp_HArray1OfPnt, TColgp_HArray2OfPnt
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_SOLID, TopAbs_VERTEX, TopAbs_SHAPE
from OCC.Core.TopoDS import TopoDS_Iterator, topods_Vertex
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepProj import BRepProj_Projection
from OCC.Core.BRepFill import BRepFill_Filling
from OCC.Core.BRepFill import BRepFill_CurveConstraint
from OCC.Core.BRepTools import breptools_Read
from OCC.Core.BRepOffset import BRepOffset_MakeOffset, BRepOffset_Skin, BRepOffset_Interval
from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_ThruSections, BRepOffsetAPI_MakeOffset, BRepOffsetAPI_MakeEvolved, BRepOffsetAPI_MakePipe, BRepOffsetAPI_MakePipeShell
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeBox
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeWire
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeFace
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_GTransform
from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCC.Core.BRepLProp import BRepLProp_SurfaceTool, BRepLProp_SLProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepBndLib import brepbndlib, brepbndlib_Add, brepbndlib_AddOBB, brepbndlib_AddOptimal
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.Bnd import Bnd_Box, Bnd_OBB
from OCC.Core.Geom import Geom_Plane, Geom_Surface, Geom_BSplineSurface
from OCC.Core.Geom import Geom_Curve, Geom_Line, Geom_Ellipse, Geom_Circle
from OCC.Core.Geom import Geom_RectangularTrimmedSurface, Geom_ToroidalSurface
from OCC.Core.GeomAPI import geomapi
from OCC.Core.GeomAPI import GeomAPI_IntCS, GeomAPI_IntSS
from OCC.Core.GeomAPI import GeomAPI_PointsToBSplineSurface
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomAPI import GeomAPI_Interpolate
from OCC.Core.GeomAbs import GeomAbs_C0, GeomAbs_C1, GeomAbs_C2
from OCC.Core.GeomAbs import GeomAbs_G1, GeomAbs_G2
from OCC.Core.GeomAbs import GeomAbs_Intersection, GeomAbs_Arc
from OCC.Core.GeomFill import GeomFill_BoundWithSurf
from OCC.Core.GeomFill import GeomFill_BSplineCurves
from OCC.Core.GeomFill import GeomFill_StretchStyle, GeomFill_CoonsStyle, GeomFill_CurvedStyle
from OCC.Core.AIS import AIS_Manipulator
from OCC.Core.V3d import V3d_SpotLight, V3d_XnegYnegZpos, V3d_XposYposZpos
from OCC.Core.Graphic3d import Graphic3d_NOM_ALUMINIUM, Graphic3d_NOM_COPPER, Graphic3d_NOM_BRASS
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_WHITE, Quantity_NOC_CORAL2, Quantity_NOC_BROWN
from OCC.Core.BRepTools import breptools_Write
from OCC.Extend.DataExchange import write_step_file, read_step_file
from OCC.Extend.DataExchange import write_iges_file, read_iges_file
from OCC.Extend.DataExchange import write_stl_file, read_stl_file
from OCC.Extend.ShapeFactory import midpoint
from OCCUtils.Topology import Topo
from OCCUtils.Topology import shapeTypeString, dumpTopology
from OCCUtils.Construct import make_box, make_face, make_line, make_wire, make_edge
from OCCUtils.Construct import make_plane, make_polygon
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import dir_to_vec, vec_to_dir


def which(program):
    """Run the Unix which command in Python."""
    import os

    def is_exe(fpath):
        """Check if file is executable."""
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, _ = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None


def _gmsh_path():
    """Find Gmsh."""

    if os.name == "nt":
        gmp = which("gmsh.exe")
    else:
        gmp = which("gmsh")
    if gmp is None:
        print(
            "Could not find Gmsh."
            + "Interactive plotting and shapes module not available."
        )
    return gmp


def sys_flush(n):
    sys.stdout.write("\r " + " / ".join(map(str, n)))
    sys.stdout.flush()


def split_filename(filename="./temp_20200408000/not_ignore.txt"):
    name = os.path.basename(filename)
    rootname, ext_name = os.path.splitext(name)
    return name, rootname


def create_tempdir(name="temp", flag=1, d="./"):
    print(datetime.date.today(), time.ctime())
    datenm = "{0:%Y%m%d}".format(datetime.date.today())
    dirnum = len(glob.glob(d + "{}_{}*/".format(name, datenm)))
    if flag == -1 or dirnum == 0:
        tmpdir = d + "{}_{}{:03}/".format(name, datenm, dirnum)
        os.makedirs(tmpdir)
        fp = open(tmpdir + "not_ignore.txt", "w")
        fp.close()
    else:
        tmpdir = d + "{}_{}{:03}/".format(name, datenm, dirnum - 1)
    return tmpdir


def create_tempnum(name, tmpdir="./", ext=".tar.gz"):
    num = len(glob.glob(tmpdir + name + "*" + ext)) + 1
    filename = '{}{}_{:03}{}'.format(tmpdir, name, num, ext)
    #print(num, filename)
    return filename


def create_tempdate(name, tmpdir="./", ext=".tar.gz"):
    print(datetime.date.today())
    datenm = "{0:%Y%m%d}".format(datetime.date.today())
    num = len(glob.glob(tmpdir + name + "_{}*".format(datenm) + ext)) + 1
    filename = '{}{}_{}{:03}{}'.format(tmpdir, name, datenm, num, ext)
    return filename


class SetDir (object):

    def __init__(self, temp=True):
        self.root_dir = os.getcwd()
        self.tempname = ""
        self.rootname = ""

        pyfile = sys.argv[0]
        self.filename = os.path.basename(pyfile)
        self.rootname, ext_name = os.path.splitext(self.filename)

        if temp == True:
            self.create_tempdir()
            self.tempname = self.tmpdir + self.rootname
            print(self.rootname)
        else:
            print(self.tmpdir)

    def init(self):
        self.tempname = self.tmpdir + self.rootname

    def create_tempdir(self, name="temp", flag=1, d="./"):
        self.tmpdir = create_tempdir(name, flag, d)
        self.tempname = self.tmpdir + self.rootname
        print(self.tmpdir)

    def create_dir(self, name="temp"):
        os.makedirs(name, exist_ok=True)
        if os.path.isdir(name):
            os.makedirs(name, exist_ok=True)
            fp = open(name + "not_ignore.txt", "w")
            fp.close()
            print("make {}".format(name))
        else:
            print("already exist {}".format(name))
        return name

    def create_dirnum(self, name="./temp", flag=+1):
        dirnum = len(glob.glob("{}_*/".format(name))) + flag
        if dirnum < 0:
            dirnum = 0
        dirname = name + "_{:03}/".format(dirnum)
        os.makedirs(dirname, exist_ok=True)
        fp = open(dirname + "not_ignore.txt", "w")
        fp.close()
        print("make {}".format(dirname))
        return dirname

    def add_tempdir(self, dirname="./", name="temp", flag=1):
        self.tmpdir = dirname
        self.tmpdir = create_tempdir(self.tmpdir + name, flag)
        self.tempname = self.tmpdir + self.rootname
        print(self.tmpdir)

    def add_dir(self, name="temp"):
        dirnum = len(glob.glob("{}/{}/".format(self.tmpdir, name)))
        if dirnum == 0:
            tmpdir = "{}/{}/".format(self.tmpdir, name)
            os.makedirs(tmpdir)
            fp = open(tmpdir + "not_ignore.txt", "w")
            fp.close()
            print("make {}".format(tmpdir))
        else:
            tmpdir = "{}/{}/".format(self.tmpdir, name)
            print("already exist {}".format(tmpdir))
        return tmpdir

    def add_dir_num(self, name="temp", flag=-1):
        if flag == -1:
            num = len(glob.glob("{}/{}_*".format(self.tmpdir, name))) + 1
        else:
            num = len(glob.glob("{}/{}_*".format(self.tmpdir, name)))
        tmpdir = "{}/{}_{:03}/".format(self.tmpdir, name, num)
        os.makedirs(tmpdir, exist_ok=True)
        fp = open(tmpdir + "not_ignore.txt", "w")
        fp.close()
        print("make {}".format(tmpdir))
        return tmpdir

    def open_filemanager(self, path="."):
        abspath = os.path.abspath(path)
        if sys.platform == "win32":
            subprocess.run('explorer.exe {}'.format(abspath))
        elif sys.platform == "linux":
            subprocess.check_call(['xdg-open', abspath])
        else:
            subprocess.run('explorer.exe {}'.format(abspath))

    def open_tempdir(self):
        self.open_filemanager(self.tmpdir)

    def open_newtempdir(self):
        self.create_tempdir("temp", -1)
        self.open_tempdir()

    def exit_app(self):
        sys.exit()


class PlotBase(SetDir):

    def __init__(self, aspect="equal", temp=True):
        if temp == True:
            SetDir.__init__(self, temp)
        self.dim = 2
        self.fig, self.axs = plt.subplots()

    def new_fig(self, aspect="equal", dim=None):
        if dim == None:
            self.new_fig(aspect=aspect, dim=self.dim)
        elif self.dim == 2:
            self.new_2Dfig(aspect=aspect)
        elif self.dim == 3:
            self.new_3Dfig(aspect=aspect)
        else:
            self.new_2Dfig(aspect=aspect)

    def new_2Dfig(self, aspect="equal"):
        self.fig, self.axs = plt.subplots()
        self.axs.set_aspect(aspect)
        self.axs.xaxis.grid()
        self.axs.yaxis.grid()

    def new_3Dfig(self, aspect="equal"):
        self.fig = plt.figure()
        self.axs = self.fig.add_subplot(111, projection='3d')
        #self.axs = self.fig.gca(projection='3d')
        # self.axs.set_aspect('equal')

        self.axs.set_xlabel('x')
        self.axs.set_ylabel('y')
        self.axs.set_zlabel('z')

        self.axs.xaxis.grid()
        self.axs.yaxis.grid()
        self.axs.zaxis.grid()

    def SavePng(self, pngname=None):
        if pngname == None:
            pngname = self.tmpdir + self.rootname + ".png"
        self.fig.savefig(pngname)

    def SavePng_Serial(self, pngname=None):
        if pngname == None:
            pngname = self.rootname
            dirname = self.tmpdir
        else:
            if os.path.dirname(pngname) == "":
                dirname = "./"
            else:
                dirname = os.path.dirname(pngname) + "/"
            basename = os.path.basename(pngname)
            pngname, extname = os.path.splitext(basename)
        pngname = create_tempnum(pngname, dirname, ".png")
        self.fig.savefig(pngname)

    def Show(self):
        try:
            plt.show()
        except AttributeError:
            pass

    def plot_close(self):
        plt.close("all")


def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)


class plot2d (PlotBase):

    def __init__(self, aspect="equal", temp=True, *args, **kwargs):
        PlotBase.__init__(self, aspect, temp, *args, **kwargs)
        self.dim = 2
        # self.new_2Dfig(aspect=aspect)
        self.new_fig(aspect=aspect)

    def add_axs(self, row=1, col=1, num=1, aspect="auto"):
        self.axs.set_axis_off()
        axs = self.fig.add_subplot(row, col, num)
        axs.set_aspect(aspect)
        axs.xaxis.grid()
        axs.yaxis.grid()
        return axs

    def add_twin(self, aspect="auto", side="right", out=0):
        axt = self.axs.twinx()
        # axt.axis("off")
        axt.set_aspect(aspect)
        axt.xaxis.grid()
        axt.yaxis.grid()
        axt.spines[side].set_position(('axes', out))
        make_patch_spines_invisible(axt)
        axt.spines[side].set_visible(True)
        return axt

    def div_axs(self):
        self.div = make_axes_locatable(self.axs)
        # self.axs.set_aspect('equal')

        self.ax_x = self.div.append_axes(
            "bottom", 1.0, pad=0.5, sharex=self.axs)
        self.ax_x.xaxis.grid(True, zorder=0)
        self.ax_x.yaxis.grid(True, zorder=0)

        self.ax_y = self.div.append_axes(
            "right", 1.0, pad=0.5, sharey=self.axs)
        self.ax_y.xaxis.grid(True, zorder=0)
        self.ax_y.yaxis.grid(True, zorder=0)

    def contourf_sub(self, mesh, func, sxy=[0, 0], pngname=None):
        self.new_fig()
        self.div_axs()
        nx, ny = mesh[0].shape
        sx, sy = sxy
        xs, xe = mesh[0][0, 0], mesh[0][0, -1]
        ys, ye = mesh[1][0, 0], mesh[1][-1, 0]
        mx = np.searchsorted(mesh[0][0, :], sx) - 1
        my = np.searchsorted(mesh[1][:, 0], sy) - 1

        self.ax_x.plot(mesh[0][mx, :], func[mx, :])
        self.ax_x.set_title("y = {:.2f}".format(sy))
        self.ax_y.plot(func[:, my], mesh[1][:, my])
        self.ax_y.set_title("x = {:.2f}".format(sx))
        im = self.axs.contourf(*mesh, func, cmap="jet")
        self.fig.colorbar(im, ax=self.axs, shrink=0.9)
        self.fig.tight_layout()
        self.SavePng(pngname)

    def contourf_tri(self, x, y, z):
        self.new_fig()
        self.axs.tricontourf(x, y, z, cmap="jet")

    def contourf_div(self, mesh, func, loc=[0, 0], txt="", title="name", pngname="./tmp/png", level=None):
        sx, sy = loc
        nx, ny = func.shape
        xs, ys = mesh[0][0, 0], mesh[1][0, 0]
        xe, ye = mesh[0][0, -1], mesh[1][-1, 0]
        dx, dy = mesh[0][0, 1] - mesh[0][0, 0], mesh[1][1, 0] - mesh[1][0, 0]
        mx, my = int((sy - ys) / dy), int((sx - xs) / dx)
        tx, ty = 1.1, 0.0

        self.new_2Dfig()
        self.div_axs()
        self.ax_x.plot(mesh[0][mx, :], func[mx, :])
        self.ax_x.set_title("y = {:.2f}".format(sy))

        self.ax_y.plot(func[:, my], mesh[1][:, my])
        self.ax_y.set_title("x = {:.2f}".format(sx))

        self.fig.text(tx, ty, txt, transform=self.ax_x.transAxes)
        im = self.axs.contourf(*mesh, func, cmap="jet", levels=level)
        self.axs.set_title(title)
        self.fig.colorbar(im, ax=self.axs, shrink=0.9)

        plt.tight_layout()
        plt.savefig(pngname + ".png")

    def contourf_div_auto(self, mesh, func, loc=[0, 0], txt="", title="name", pngname="./tmp/png", level=None):
        sx, sy = loc
        nx, ny = func.shape
        xs, ys = mesh[0][0, 0], mesh[1][0, 0]
        xe, ye = mesh[0][0, -1], mesh[1][-1, 0]
        dx, dy = mesh[0][0, 1] - mesh[0][0, 0], mesh[1][1, 0] - mesh[1][0, 0]
        mx, my = int((sy - ys) / dy), int((sx - xs) / dx)
        tx, ty = 1.1, 0.0

        self.new_2Dfig()
        self.div_axs()
        self.axs.set_aspect('auto')
        self.ax_x.plot(mesh[0][mx, :], func[mx, :])
        self.ax_x.set_title("y = {:.2f}".format(sy))

        self.ax_y.plot(func[:, my], mesh[1][:, my])
        self.ax_y.set_title("x = {:.2f}".format(sx))

        self.fig.text(tx, ty, txt, transform=self.ax_x.transAxes)
        im = self.axs.contourf(*mesh, func, cmap="jet", levels=level)
        self.axs.set_title(title)
        self.fig.colorbar(im, ax=self.axs, shrink=0.9)

        plt.tight_layout()
        plt.savefig(pngname + ".png")


class plot3d (PlotBase):

    def __init__(self, aspect="equal", *args, **kwargs):
        PlotBase.__init__(self, *args, **kwargs)
        self.dim = 3
        self.new_fig()

    def set_axes_equal(self):
        '''
        Make axes of 3D plot have equal scale so that spheres appear as spheres,
        cubes as cubes, etc..  This is one possible solution to Matplotlib's
        ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

        Input
          ax: a matplotlib axis, e.g., as output from plt.gca().
        '''

        x_limits = self.axs.get_xlim3d()
        y_limits = self.axs.get_ylim3d()
        z_limits = self.axs.get_zlim3d()

        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])

        x_middle = np.mean(x_limits)
        y_middle = np.mean(y_limits)
        z_middle = np.mean(z_limits)

        # The plot bounding box is a sphere in the sense of the infinity
        # norm, hence I call half the max range the plot radius.
        plot_radius = 0.5 * max([x_range, y_range, z_range])

        self.axs.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
        self.axs.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
        self.axs.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

    def plot_ball(self, rxyz=[1, 1, 1]):
        u = np.linspace(0, 1, 10) * 2 * np.pi
        v = np.linspace(0, 1, 10) * np.pi
        uu, vv = np.meshgrid(u, v)
        x = rxyz[0] * np.cos(uu) * np.sin(vv)
        y = rxyz[1] * np.sin(uu) * np.sin(vv)
        z = rxyz[2] * np.cos(vv)

        self.axs.plot_wireframe(x, y, z)
        self.set_axes_equal()
        #self.axs.set_xlim3d(-10, 10)
        #self.axs.set_ylim3d(-10, 10)
        #self.axs.set_zlim3d(-10, 10)


def rotate_xyz(axs=gp_Ax3(), deg=0.0, xyz="x"):
    if xyz == "x":
        ax1 = gp_Ax1(axs.Location(), axs.XDirection())
    elif xyz == "y":
        ax1 = gp_Ax1(axs.Location(), axs.YDirection())
    elif xyz == "z":
        ax1 = gp_Ax1(axs.Location(), axs.Direction())
    else:
        ax1 = gp_Ax1(axs.Location(), axs.Direction())
    axs.Rotate(ax1, np.deg2rad(deg))


def pnt_from_axs(axs=gp_Ax3(), length=100):
    vec = point_to_vector(axs.Location()) + \
        dir_to_vec(axs.Direction()) * length
    return vector_to_point(vec)


def line_from_axs(axs=gp_Ax3(), length=100):
    return make_edge(axs.Location(), pnt_from_axs(axs, length))


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


def rotate_axs(axs=gp_Ax3(), deg=0.0, idx="x"):
    ax = axs.Axis()
    if idx == "x":
        ax.SetDirection(axs.XDirection())
    elif idx == "y":
        ax.SetDirection(axs.YDirection())
    elif idx == "z":
        ax.SetDirection(axs.Direction())
    else:
        ax.SetDirection(axs.Direction())
    axs.Rotate(ax, np.deg2rad(deg))


def rot_axs(axis=gp_Ax3(), pxyz=[0, 0, 0], rxyz=[0, 0, 0]):
    axs = gp_Ax3(gp_Pnt(*pxyz), gp_Dir(0, 0, 1))
    ax1 = gp_Ax1(axis.Location(), axis.XDirection())
    ax2 = gp_Ax1(axis.Location(), axis.YDirection())
    ax3 = gp_Ax1(axis.Location(), axis.Direction())
    axs.Rotate(ax1, np.deg2rad(rxyz[0]))
    axs.Rotate(ax2, np.deg2rad(rxyz[1]))
    axs.Rotate(ax3, np.deg2rad(rxyz[2]))
    trsf = set_trf(gp_Ax3(), axs)
    axis.Transform(trsf)


def trf_axs(axs=gp_Ax3(), pxyz=[0, 0, 0], rxyz=[0, 0, 0]):
    rotate_axs(axs, rxyz[0], "x")
    rotate_axs(axs, rxyz[1], "y")
    rotate_axs(axs, rxyz[2], "z")
    axs.SetLocation(gp_Pnt(*pxyz))


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


def spl_face(px, py, pz, axs=gp_Ax3()):
    nx, ny = px.shape
    pnt_2d = TColgp_Array2OfPnt(1, nx, 1, ny)
    for row in range(pnt_2d.LowerRow(), pnt_2d.UpperRow() + 1):
        for col in range(pnt_2d.LowerCol(), pnt_2d.UpperCol() + 1):
            i, j = row - 1, col - 1
            pnt = gp_Pnt(px[i, j], py[i, j], pz[i, j])
            pnt_2d.SetValue(row, col, pnt)
            #print (i, j, px[i, j], py[i, j], pz[i, j])

    api = GeomAPI_PointsToBSplineSurface(pnt_2d, 3, 8, GeomAbs_G2, 0.001)
    api.Interpolate(pnt_2d)
    #surface = BRepBuilderAPI_MakeFace(curve, 1e-6)
    # return surface.Face()
    face = BRepBuilderAPI_MakeFace(api.Surface(), 1e-6).Face()
    face.Location(set_loc(gp_Ax3(), axs))
    return face


def spl_curv(px, py, pz):
    num = px.size
    pts = []
    p_array = TColgp_Array1OfPnt(1, num)
    for idx, t in enumerate(px):
        x = px[idx]
        y = py[idx]
        z = pz[idx]
        pnt = gp_Pnt(x, y, z)
        pts.append(pnt)
        p_array.SetValue(idx + 1, pnt)
    api = GeomAPI_PointsToBSpline(p_array)
    return p_array, api.Curve()


def spl_curv_pts(pts=[gp_Pnt()]):
    num = len(pts)
    p_array = TColgp_Array1OfPnt(1, num)
    for idx, pnt in enumerate(pts):
        p_array.SetValue(idx + 1, pnt)
    api = GeomAPI_PointsToBSpline(p_array)
    return p_array, api.Curve()


def surf_spl_pcd(px, py, pz):
    nx, ny = px.shape
    pnt_2d = TColgp_Array2OfPnt(1, nx, 1, ny)
    for row in range(pnt_2d.LowerRow(), pnt_2d.UpperRow() + 1):
        for col in range(pnt_2d.LowerCol(), pnt_2d.UpperCol() + 1):
            i, j = row - 1, col - 1
            pnt = gp_Pnt(px[i, j], py[i, j], pz[i, j])
            pnt_2d.SetValue(row, col, pnt)

    curv = GeomAPI_PointsToBSplineSurface(
        pnt_2d, 3, 8, GeomAbs_G2, 0.001).Surface()
    surf = BRepBuilderAPI_MakeFace(curv, 1e-6).Face()
    return surf, pnt_2d


def curvature(px, r, s):
    """( x + sx )**2 / 2*rx + ( y + sy )**2 / 2*ry"""
    if (r == 0):
        py = np.zeros_like(px + s)
    else:
        py = (px + s)**2 / (2 * r)
    return py


def pnt_to_xyz(p):
    return p.X(), p.Y(), p.Z()


def float_to_string(number):
    if number == 0 or abs(np.log10(abs(number))) < 100:
        return ' {: 0.10E}'.format(number)
    else:
        return ' {: 0.10E}'.format(number).replace('E', '')


def get_deg(axs, vec):
    vx = dir_to_vec(axs.XDirection())
    vy = dir_to_vec(axs.YDirection())
    vz = dir_to_vec(axs.Direction())
    pln_x = Geom_Plane(axs.Location(), axs.YDirection())
    pln_y = Geom_Plane(axs.Location(), axs.XDirection())
    vec_p = gp_Pnt((gp_Vec(axs.Location().XYZ()) + vec).XYZ())
    pnt_x = GeomAPI_ProjectPointOnSurf(vec_p, pln_x).Point(1)
    pnt_y = GeomAPI_ProjectPointOnSurf(vec_p, pln_y).Point(1)
    vec_x = gp_Vec(axs.Location(), pnt_x)
    vec_y = gp_Vec(axs.Location(), pnt_y)
    deg_x = vec_x.AngleWithRef(vz, vy)
    deg_y = vec_y.AngleWithRef(vz, vx)
    print(np.rad2deg(deg_x), np.rad2deg(deg_y))
    return deg_x, deg_y


def get_axs(filename, ax=gp_Ax3()):
    dat = np.loadtxt(filename, skiprows=2)
    pnt = gp_Pnt(*dat[0])
    d_x = gp_Dir(*dat[1])
    d_y = gp_Dir(*dat[2])
    d_z = d_x.Crossed(d_y)
    axs = gp_Ax3(pnt, d_z, d_x)
    trf = gp_Trsf()
    trf.SetTransformation(ax, gp_Ax3())
    axs.Transform(trf)
    return axs


def generate_rim_pcd(axs, pts, filename="pln.rim", name="name", nxy=5):
    pnt = axs.Location()
    trf = gp_Trsf()
    trf.SetTransformation(gp_Ax3(), axs)
    px, py = [], []
    for i in range(len(pts)):
        i0, i1 = i, (i + 1) % len(pts)
        p0, p1 = pts[i0].Transformed(trf), pts[i1].Transformed(trf)
        p_x = np.delete(np.linspace(p0.X(), p1.X(), nxy), -1)
        p_y = np.delete(np.linspace(p0.Y(), p1.Y(), nxy), -1)
        px.extend(p_x), py.extend(p_y)


def occ_to_grasp_rim(axs, pts, filename="pln.rim", name="name", nxy=5):
    pnt = axs.Location()
    trf = gp_Trsf()
    trf.SetTransformation(gp_Ax3(), axs)
    px, py = [], []
    for i in range(len(pts)):
        i0, i1 = i, (i + 1) % len(pts)
        p0, p1 = pts[i0].Transformed(trf), pts[i1].Transformed(trf)
        p_x = np.delete(np.linspace(p0.X(), p1.X(), nxy), -1)
        p_y = np.delete(np.linspace(p0.Y(), p1.Y(), nxy), -1)
        px.extend(p_x), py.extend(p_y)

    fp = open(filename, "w")
    fp.write(' {:s}\n'.format(name))
    fp.write('{:12d}{:12d}{:12d}\n'.format(len(px), 1, 1))
    #fp.write(' {:s}\n'.format("mm"))
    for i in range(len(px)):
        data = [px[i], py[i]]
        fp.write(''.join([float_to_string(val) for val in data]) + '\n')


def occ_to_grasp_cor(axs, name="name", filename="pln.cor"):
    pnt = axs.Location()
    v_x = axs.XDirection()
    v_y = axs.YDirection()
    fp = open(filename, "w")
    fp.write(' {:s}\n'.format(name))
    fp.write(' {:s}\n'.format("mm"))
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(pnt)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_x)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_y)]) + '\n')
    fp.close()


def occ_to_grasp_cor_ref(axs, ref=gp_Ax3(), name="name", filename="pln.cor"):
    trf = gp_Trsf()
    trf.SetTransformation(gp_Ax3(), ref)
    # trf.Invert()
    ax1 = axs.Transformed(trf)
    pnt = ax1.Location()
    v_x = ax1.XDirection()
    v_y = ax1.YDirection()
    fp = open(filename, "w")
    fp.write(' {:s}\n'.format(name))
    fp.write(' {:s}\n'.format("mm"))
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(pnt)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_x)]) + '\n')
    fp.write(''.join([float_to_string(v) for v in pnt_to_xyz(v_y)]) + '\n')
    fp.close()


def grasp_sfc(mesh, surf, sfc_file="surf.sfc"):
    fp = open(sfc_file, "w")
    ny, nx = surf.shape
    xs, xe = mesh[0][0, 0], mesh[0][0, -1]
    ys, ye = mesh[1][0, 0], mesh[1][-1, 0]
    fp.write(" {} data \n".format(sfc_file))
    fp.write(" {:.2e} {:.2e} {:.2e} {:.2e}\n".format(xs, ys, xe, ye))
    fp.write(" {:d} {:d}\n".format(nx, ny))
    for ix in range(nx):
        for iy in range(ny):
            fp.write(" {:.5e} ".format(surf[iy, ix]))
        fp.write("\n")
    fp.close()


def get_boundxyz_pts(pts, axs=gp_Ax3()):
    poly = make_polygon(pts, closed=True)
    poly.Location(set_loc(gp_Ax3(), axs))
    n_sided = BRepFill_Filling()
    for e in Topo(poly).edges():
        n_sided.Add(e, GeomAbs_C0)
    n_sided.Build()

    face = n_sided.Face()
    solid = BRepOffset_MakeOffset(
        face, 10.0, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
    shape = solid.Shape()

    bbox = Bnd_Box()
    bbox.Set(axs.Location(), axs.Direction())
    brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax]


def get_boundxyz_rim(rim, axs=gp_Ax3()):
    n_sided = BRepFill_Filling()
    for e in Topo(rim).edges():
        n_sided.Add(e, GeomAbs_C0)
    n_sided.Build()

    face = n_sided.Face()
    solid = BRepOffset_MakeOffset(
        face, 10.0, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
    shape = solid.Shape()

    bbox = Bnd_Box()
    bbox.Set(axs.Location(), axs.Direction())
    brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax]


def get_boundxyz_face(face, axs=gp_Ax3()):
    solid = BRepOffset_MakeOffset(
        face, 10.0, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
    shape = solid.Shape()

    bbox = Bnd_Box()
    bbox.Set(axs.Location(), axs.XDirection())
    brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax]


def get_aligned_boundingbox_ratio(shape, tol=1e-6, optimal_BB=True, ratio=1):
    """ return the bounding box of the TopoDS_Shape `shape`

    Parameters
    ----------

    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from

    tol: float
        tolerance of the computed boundingbox

    use_triangulation : bool, True by default
        This makes the computation more accurate

    ratio : float, 1.0 by default.

    Returns
    -------
        if `as_pnt` is True, return a tuple of gp_Pnt instances
         for the lower and another for the upper X,Y,Z values representing the bounding box

        if `as_pnt` is False, return a tuple of lower and then upper X,Y,Z values
         representing the bounding box
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)

    # note: useTriangulation is True by default, we set it explicitely, but t's not necessary
    if optimal_BB:
        use_triangulation = True
        use_shapetolerance = True
        brepbndlib_AddOptimal(
            shape, bbox, use_triangulation, use_shapetolerance)
    else:
        brepbndlib_Add(shape, bbox)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    dx, mx = (xmax - xmin) * ratio, (xmax + xmin) / 2
    dy, my = (ymax - ymin) * ratio, (ymax + ymin) / 2
    dz, mz = (zmax - zmin) * ratio, (zmax + zmin) / 2
    x0, x1 = mx - dx / 2, mx + dx / 2
    y0, y1 = my - dy / 2, my + dy / 2
    z0, z1 = mz - dz / 2, mz + dz / 2
    corner1 = gp_Pnt(x0, y0, z0)
    corner2 = gp_Pnt(x1, y1, z1)
    center = midpoint(corner1, corner2)

    rim0 = make_polygon(
        [gp_Pnt(x0, y0, z0),
         gp_Pnt(x1, y0, z0),
         gp_Pnt(x1, y1, z0),
         gp_Pnt(x0, y1, z0)],
        closed=True
    )

    rim1 = make_polygon(
        [gp_Pnt(x0, y0, z1),
         gp_Pnt(x1, y0, z1),
         gp_Pnt(x1, y1, z1),
         gp_Pnt(x0, y1, z1)],
        closed=True
    )
    api = BRepOffsetAPI_ThruSections(True, False, 1.0E-9)
    api.AddWire(rim0)
    api.AddWire(rim1)
    box_shp = api.Shape()
    #box_shp = BRepPrimAPI_MakeBox(corner1, corner2).Shape()
    return center, [dx, dy, dz], box_shp


def get_oriented_boundingbox_ratio(shape, optimal_OBB=True, ratio=1.0):
    """ return the oriented bounding box of the TopoDS_Shape `shape`

    Parameters
    ----------

    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from

    optimal_OBB : bool, True by default. If set to True, compute the
        optimal (i.e. the smallest oriented bounding box). 
        Optimal OBB is a bit longer.

    ratio : float, 1.0 by default.

    Returns
    -------
        a list with center, x, y and z sizes

        a shape
    """
    obb = Bnd_OBB()
    if optimal_OBB:
        is_triangulationUsed = True
        is_optimal = True
        is_shapeToleranceUsed = False
        brepbndlib_AddOBB(shape, obb, is_triangulationUsed,
                          is_optimal, is_shapeToleranceUsed)
    else:
        brepbndlib_AddOBB(shape, obb)

    # converts the bounding box to a shape
    aBaryCenter = obb.Center()
    aXDir = obb.XDirection()
    aYDir = obb.YDirection()
    aZDir = obb.ZDirection()
    aHalfX = obb.XHSize()
    aHalfY = obb.YHSize()
    aHalfZ = obb.ZHSize()
    dx = aHalfX * ratio
    dy = aHalfY * ratio
    dz = aHalfZ * ratio

    ax = gp_XYZ(aXDir.X(), aXDir.Y(), aXDir.Z())
    ay = gp_XYZ(aYDir.X(), aYDir.Y(), aYDir.Z())
    az = gp_XYZ(aZDir.X(), aZDir.Y(), aZDir.Z())
    p = gp_Pnt(aBaryCenter.X(), aBaryCenter.Y(), aBaryCenter.Z())
    anAxes = gp_Ax2(p, gp_Dir(aZDir), gp_Dir(aXDir))
    anAxes.SetLocation(gp_Pnt(p.XYZ() - ax * dx - ay * dy - az * dz))
    aBox = BRepPrimAPI_MakeBox(anAxes, 2.0 * dx, 2.0 * dy, 2.0 * dz).Shape()
    return aBaryCenter, [dx, dy, dz], aBox


def face_mesh_triangle(comp=TopoDS_Shape(), isR=0.1, thA=0.1):
    # Mesh the shape
    BRepMesh_IncrementalMesh(comp, isR, True, thA, True)
    bild1 = BRep_Builder()
    comp1 = TopoDS_Compound()
    bild1.MakeCompound(comp1)
    bt = BRep_Tool()
    ex = TopExp_Explorer(comp, TopAbs_FACE)
    while ex.More():
        face = topods_Face(ex.Current())
        location = TopLoc_Location()
        facing = bt.Triangulation(face, location)
        tab = facing.Nodes()
        tri = facing.Triangles()
        print(facing.NbTriangles(), facing.NbNodes())
        for i in range(1, facing.NbTriangles() + 1):
            trian = tri.Value(i)
            index1, index2, index3 = trian.Get()
            pnts = [tab.Value(index1), tab.Value(index2), tab.Value(index3)]
            poly = make_polygon(pnts, closed=True)
            ftri = make_face(poly, True)
            bild1.Add(comp1, poly)
            bild1.Add(comp1, ftri)
            # for j in range(1, 4):
            #    if j == 1:
            #        m = index1
            #        n = index2
            #    elif j == 2:
            #        n = index3
            #    elif j == 3:
            #        m = index2
            #    me = BRepBuilderAPI_MakeEdge(tab.Value(m), tab.Value(n))
            #    if me.IsDone():
            #        bild1.Add(comp1, me.Edge())
        ex.Next()
    return comp1


def check_callable(_callable):
    if not callable(_callable):
        raise AssertionError("The function supplied is not callable")


log = logging.getLogger(__name__)
from PyQt5 import QtCore, QtGui, QtOpenGL, QtWidgets
from OCC import VERSION
from OCC.Display.backend import load_backend, get_qt_modules
from OCC.Display.OCCViewer import OffscreenRenderer
# https://github.com/tpaviot/pythonocc-core/issues/999


class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, backend_str=None, *args):
        used_backend = load_backend(backend_str)
        log.info("GUI backend set to: %s", used_backend)
        from OCC.Display.qtDisplay import qtViewer3d, qtBaseViewer

        # following couple of lines is a tweak to enable ipython --gui='qt'
        # checks if QApplication already exists
        self.app = QtWidgets.QApplication.instance()
        if not self.app:  # create QApplication if it doesnt exist
            self.app = QtWidgets.QApplication(sys.argv)

        QtWidgets.QMainWindow.__init__(self, *args)
        self.canvas = qtViewer3d(self)
        self.setWindowTitle(
            "pythonOCC-%s 3d viewer ('%s' backend)" % (VERSION, used_backend))
        self.setCentralWidget(self.canvas)
        if sys.platform != 'darwin':
            self.menu_bar = self.menuBar()
        else:
            # create a parentless menubar
            # see: http://stackoverflow.com/questions/11375176/qmenubar-and-qmenu-doesnt-show-in-mac-os-x?lq=1
            # noticeable is that the menu ( alas ) is created in the
            # top-left of the screen, just next to the apple icon
            # still does ugly things like showing the "Python" menu in bold
            self.menu_bar = QtWidgets.QMenuBar()
        self._menus = {}
        self._menu_methods = {}
        # place the window in the center of the screen, at half the
        # screen size
        self.centerOnScreen()

    def centerOnScreen(self):
        '''Centers the window on the screen.'''
        reso = QtWidgets.QApplication.desktop().screenGeometry()
        frme = self.frameSize()
        sx = int((reso.width() - frme.width()) / 2)
        sy = int((reso.height() - frme.height()) / 2)
        self.move(sx, sy)

    def _add_menu(self, menu_name):
        _menu = self.menu_bar.addMenu("&" + menu_name)
        self._menus[menu_name] = _menu

    def _add_function_to_menu(self, menu_name, _callable):
        check_callable(_callable)
        try:
            _action = QtWidgets.QAction(
                _callable.__name__.replace('_', ' ').lower(), self)
            # if not, the "exit" action is now shown...
            _action.setMenuRole(QtWidgets.QAction.NoRole)
            _action.triggered.connect(_callable)
            self._menus[menu_name].addAction(_action)
        except KeyError:
            raise ValueError('the menu item %s does not exist' % menu_name)


class init_QDisplay (MainWindow):

    def __init__(self,
                 backend_str=None,
                 size=(1024, 768),
                 display_triedron=True,
                 background_gradient_color1=[206, 215, 222],
                 background_gradient_color2=[128, 128, 128]):
        MainWindow.__init__(self, backend_str)

        self.resize(size[0] - 1, size[1] - 1)
        self.show()
        self.canvas.InitDriver()
        self.resize(size[0], size[1])
        self.canvas.qApp = self.app
        self.display = self.canvas._display

        if display_triedron:
            self.display.display_triedron()

        if background_gradient_color1 and background_gradient_color2:
            # background gradient
            self.display.set_bg_gradient_color(
                background_gradient_color1, background_gradient_color2)

    def add_menu(self, *args, **kwargs):
        self._add_menu(*args, **kwargs)

    def add_menu_shortcut(self, menu_name):
        _menu = self.menu_bar.addMenu("&" + menu_name)
        self._menus[menu_name] = _menu

    def add_function(self, *args, **kwargs):
        self._add_function_to_menu(*args, **kwargs)

    def start_display(self):
        # make the application float to the top
        self.raise_()
        self.app.exec_()


class GenCompound (object):

    def __init__(self):
        self.builder = BRep_Builder()
        self.compound = TopoDS_Compound()
        self.builder.MakeCompound(self.compound)

    def add_shapes(self, shps=[]):
        for shp in shps:
            self.builder.Add(self.compound, shp)


class Viewer (object):

    def __init__(self, disp=True):
        if disp == True:
            from OCC.Display.qtDisplay import qtViewer3d
            #self.app = self.get_app()
            #self.wi = self.app.topLevelWidgets()[0]
            self.vi = self.findChild(qtViewer3d, "qt_viewer_3d")
            #self.vi = self.wi.findChild(qtViewer3d, "qt_viewer_3d")
        self.selected_shape = []

    def get_app(self):
        app = QApplication.instance()
        #app = qApp
        # checks if QApplication already exists
        if not app:
            app = QApplication(sys.argv)
        return app

    def on_select(self):
        self.vi.sig_topods_selected.connect(self._on_select)

    def clear_selected(self):
        self.selected_shape = []

    def _on_select(self, shapes):
        """
        Parameters
        ----------
        shape : TopoDS_Shape
        """
        for shape in shapes:
            print()
            print(shape.Location().Transformation())
            # print(shape.Location().Transformation().TranslationPart().Coord())
            self.selected_shape.append(shape)
            self.DumpTop(shape)

    def make_comp_selcted(self):
        bild = BRep_Builder()
        comp = TopoDS_Compound()
        bild.MakeCompound(comp)
        for shp in self.selected_shape:
            print(shp)
            bild.Add(comp, shp)
        return comp

    def DumpJson(self, shp):
        jsonname = create_tempnum(self.rootname, self.tmpdir, ".json")
        fp = open(jsonname, "w")
        shp.DumpJson(fp)
        fp.close()

    def DumpTop(self, shape, level=0):
        """
        Print the details of an object from the top down
        """
        brt = BRep_Tool()
        s = shape.ShapeType()
        if s == TopAbs_VERTEX:
            pnt = brt.Pnt(topods_Vertex(shape))
            dmp = " " * level
            dmp += "%s - " % shapeTypeString(shape)
            dmp += "%.5e %.5e %.5e" % (pnt.X(), pnt.Y(), pnt.Z())
            print(dmp)
        else:
            dmp = " " * level
            dmp += shapeTypeString(shape)
            print(dmp)
        it = TopoDS_Iterator(shape)
        while it.More():
            shp = it.Value()
            it.Next()
            self.DumpTop(shp, level + 1)


class OCCApp(plot2d, init_QDisplay, Viewer):

    def __init__(self, temp=True, disp=True, touch=False):
        plot2d.__init__(self, temp=temp)
        self.base_axs = gp_Ax3()
        self.disp = disp
        self.touch = touch
        self.colors = ["BLUE1", "RED", "GREEN",
                       "YELLOW", "BLACK", "WHITE", "BROWN"]

        # OCC Viewer
        if disp == True:
            #self.display, self.start_display, self.add_menu, self.add_function, self.wi = init_qtdisplay()
            init_QDisplay.__init__(self)
            if touch == True:
                Viewer.__init__(self, disp=True)
                self.on_select()

            self.SaveMenu()
            self.ViewMenu()
            self.SelectMenu()
            self.SelectMesh()
        else:
            Viewer.__init__(self, disp=False)

        # GMSH
        self.gmsh = _gmsh_path()

    def AddManipulator(self):
        self.manip = AIS_Manipulator(self.base_axs.Ax2())
        ais_shp = self.display.DisplayShape(
            self.base_axs.Location(),
            update=True
        )
        self.manip.Attach(ais_shp)

    def SaveMenu(self):
        self.add_menu("File")
        self.add_function("File", self.import_cadfile)
        self.add_function("File", self.export_cap)
        if self.touch == True:
            self.add_function("File", self.export_stp_selected)
            self.add_function("File", self.export_stl_selected)
            self.add_function("File", self.export_igs_selected)
            self.add_function("File", self.export_brep_selected)
            self.add_function("File", self.clear_selected)
        self.add_function("File", self.open_newtempdir)
        self.add_function("File", self.open_tempdir)
        self.add_function("File", self.exit_win)

    def ViewMenu(self):
        self.add_menu("View")
        self.add_function("View", self.display.FitAll)
        self.add_function("View", self.display.View_Top)  # XY-Plane(+)
        self.add_function("View", self.display.View_Bottom)  # XY-Plane(-)
        self.add_function("View", self.display.View_Rear)  # XZ-Plane(+)
        self.add_function("View", self.display.View_Front)  # XZ-Plane(-)
        self.add_function("View", self.display.View_Right)  # YZ-Plane(+)
        self.add_function("View", self.display.View_Left)  # YZ-Plane(-)
        self.add_function("View", self.view_xaxis)
        self.add_function("View", self.view_yaxis)
        self.add_function("View", self.view_zaxis)
        self.add_function("View", self.ray_tracing_mode)
        self.add_function("View", self.display.SetRasterizationMode)

    def SelectMenu(self):
        self.add_menu("Select")
        self.add_function("Select", self.display.SetSelectionModeVertex)
        self.add_function("Select", self.display.SetSelectionModeEdge)
        self.add_function("Select", self.display.SetSelectionModeFace)
        self.add_function("Select", self.SetSelectionModeShape)
        self.add_function("Select", self.display.SetSelectionModeNeutral)

    def SelectMesh(self):
        self.add_menu("Mesh")
        self.add_function("Mesh", self.gen_aligned_bounded_box)
        self.add_function("Mesh", self.gen_oriented_bounded_box)
        self.add_function("Mesh", self.gen_mesh_face)

    def SetSelectionModeShape(self):
        self.display.SetSelectionMode(TopAbs_SHAPE)

    def view_xaxis(self):
        self.display.View.Rotate(0, np.deg2rad(15), 0,
                                 0, 1, 0,
                                 True)

    def view_yaxis(self):
        self.display.View.Rotate(np.deg2rad(15), 0, 0,
                                 1, 0, 0,
                                 True)

    def view_zaxis(self):
        self.display.View.Rotate(0, 0, np.deg2rad(15),
                                 0, 0, 1,
                                 True)

    def gen_mesh_face(self):
        self.export_cap()
        comp = self.make_comp_selcted()
        mesh = face_mesh_triangle(comp, 0.1, 0.1)
        self.display.DisplayShape(mesh, update=True, transparency=0.9)
        self.export_cap()

    def gen_aligned_bounded_box(self):
        comp = self.make_comp_selcted()
        c, dxyz, box = get_aligned_boundingbox_ratio(comp, ratio=1.0)
        if self.disp == True:
            self.display.DisplayShape(box, transparency=0.9, update=True)
            self.export_cap()
        return c, dxyz, box

    def gen_oriented_bounded_box(self):
        comp = self.make_comp_selcted()
        c, dxyz, box = get_oriented_boundingbox_ratio(comp, ratio=1.0)
        if self.disp == True:
            self.display.DisplayShape(box, transparency=0.9, update=True)
            self.export_cap()
        return c, dxyz, box

    def ray_tracing_mode(self):
        # create one spotlight
        spot_light = V3d_SpotLight(
            gp_Pnt(-1000, -1000, 1000),
            V3d_XnegYnegZpos,
            Quantity_Color(Quantity_NOC_WHITE)
        )
        # display the spotlight in rasterized mode
        self.display.Viewer.AddLight(spot_light)
        self.display.View.SetLightOn()
        self.display.SetRaytracingMode(depth=8)

        # pythonocc-core=7.4.0
        # TKOpenGl | Type: Error | ID: 0 | Severity: High | Message:
        # Ray-tracing requires OpenGL 4.0+ or GL_ARB_texture_buffer_object_rgb32 extension

        # pythonocc-core=7.4.1
        # RuntimeError: Aspect_GraphicDeviceDefinitionErrorOpenGl_Window::CreateWindow: SetPixelFormat failed.
        # Error code: 2000 raised from method Init of class Display3d

    def import_geofile(self, geofile, tol=0.1):
        # msh1, msh2, msh22, msh3, msh4, msh40, msh41, msh,
        # unv, vtk, wrl, mail, stl, p3d, mesh, bdf, cgns, med,
        # diff, ir3, inp, ply2, celum, su2, x3d, dat, neu, m, key
        geo_dir = os.path.dirname(geofile)
        geo_base = os.path.basename(geofile)
        geo_name = create_tempnum(self.rootname, self.tmpdir, ".geo")
        geo_file = os.path.basename(geo_name)
        shutil.copyfile(geofile, geo_name)
        stl_name, _ = os.path.splitext(self.tmpdir + geo_file)
        stl_name += ".stl"
        stl_name = os.path.abspath(stl_name)
        txt_name, _ = os.path.splitext(self.tmpdir + geo_file)
        txt_name += "_gmsh.txt"
        txt_name = os.path.abspath(txt_name)

        gmsh_run = self.gmsh
        gmsh_run += " -tol {:.2f}".format(tol)
        gmsh_run += " " + geo_base
        gmsh_run += " -3 -o"
        gmsh_run += " {} -format stl".format(stl_name)
        gmsh_run += " -log {}".format(txt_name)

        os.chdir(geo_dir)
        gmsh_success = os.system(gmsh_run)
        os.chdir(self.root_dir)
        return stl_name

    def read_cadfile_axs(self, fileName, axs=gp_Ax1()):
        surf = self.read_cadfile(fileName, disp=False)
        trf = gp_Trsf()
        trf.SetTransformation(axs, gp_Ax3())
        surf.Location(TopLoc_Location(trf))
        return surf

    def read_cadfile(self, fileName, disp=True, col=None, trs=None):
        filesize = os.path.getsize(fileName)
        print(fileName, filesize / 1000, " kB")
        base_dir = os.path.dirname(fileName)
        basename = os.path.basename(fileName)
        rootname, extname = os.path.splitext(fileName)
        if extname in [".stp", ".step"]:
            shpe = read_step_file(fileName)
        elif extname in [".igs", ".iges"]:
            shpe = read_iges_file(fileName)
        elif extname in [".stl"]:
            shpe = read_stl_file(fileName)
        elif extname in [".brep"]:
            shpe = TopoDS_Shape()
            builder = BRep_Builder()
            breptools_Read(shpe, fileName, builder)
        elif extname in [".geo"]:
            stlfile = self.import_geofile(fileName, 0.1)
            shpe = read_stl_file(stlfile)
        else:
            print("Incorrect file index")
            # sys.exit(0)

        if disp == True:
            self.display.DisplayShape(
                shpe, update=True, color=col, transparency=trs)
        return shpe

    def import_cadfile(self):
        options = QFileDialog.Options()
        fileNames, _ = QFileDialog.getOpenFileNames(self, 'QFileDialog.getOpenFileName()', '',
                                                    'CAD files (*.stp *.step *.stl *.igs *.iges, *.brep. *.geo)',
                                                    options=options)
        for fileName in fileNames:
            print(fileName)
            self.read_cadfile(fileName, disp=True, trs=None)
            self.export_cap()

    def export_stl(self, shp, filename=None):
        if filename == None:
            filename = create_tempnum(self.rootname, self.tmpdir, ".stl")
        write_stl_file(shp, filename)

    def export_stp(self, shp, filename=None):
        if filename == None:
            filename = create_tempnum(self.rootname, self.tmpdir, ".stp")
        write_step_file(shp, filename)

    def export_igs(self, shp, filename=None):
        if filename == None:
            filename = create_tempnum(self.rootname, self.tmpdir, ".igs")
        write_iges_file(shp, filename)

    def export_cap_name(self, pngname=None):
        if pngname == None:
            pngname = create_tempnum(self.rootname, self.tmpdir, ".png")
        self.display.View.Dump(pngname)

    def export_cap(self):
        pngname = create_tempnum(self.rootname, self.tmpdir, ".png")
        self.display.View.Dump(pngname)

    def export_stp_selected(self):
        comp = self.make_comp_selcted()
        self.export_stp(comp)

    def export_stl_selected(self):
        comp = self.make_comp_selcted()
        stlname = create_tempnum(self.rootname, self.tmpdir, ".stl")
        write_stl_file(comp, stlname)

    def export_igs_selected(self):
        comp = self.make_comp_selcted()
        igsname = create_tempnum(self.rootname, self.tmpdir, ".igs")
        write_iges_file(comp, igsname)

    def export_brep_selected(self):
        comp = self.make_comp_selcted()
        brepname = create_tempnum(self.rootname, self.tmpdir, ".brep")
        breptools_Write(comp, brepname)

    def save_view(self, num="0"):
        self.display.View_Top()
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_XY" + num + ".png")

        self.display.View_Rear()
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_XZ" + num + ".png")

        self.display.View_Right()
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_YZ" + num + ".png")

        self.display.View.SetProj(V3d_XposYposZpos)
        self.display.FitAll()
        self.display.View.Dump(self.tempname + "_XYZ" + num + ".png")

    def exit_win(self):
        self.close()

    def ShowOCC(self):
        self.display.FitAll()
        self.display.View.Dump(self.tempname + ".png")
        self.start_display()


class plotocc (OCCApp):

    def __init__(self, temp=True, disp=True, touch=False):
        OCCApp.__init__(self, temp, disp, touch)

        # self._key_map = {ord('W'): self._display.SetModeWireFrame,
        #                  ord('S'): self._display.SetModeShaded,
        #                  ord('A'): self._display.EnableAntiAliasing,
        #                  ord('B'): self._display.DisableAntiAliasing,
        #                  ord('H'): self._display.SetModeHLR,
        #                  ord('F'): self._display.FitAll,
        #                  ord('G'): self._display.SetSelectionMode}

        # def keyPressEvent(self, event):
        #     code = event.key()
        #     if code in self._key_map:
        #         self._key_map[code]()
        #     elif code in range(256):
        #         log.info('key: "%s"(code %i) not mapped to any function' % (chr(code), code))
        #     else:
        #         log.info('key: code %i not mapped to any function' % code)

    def reload_app(self, temp=True, disp=True, touch=False):
        OCCApp.__init__(self, temp, disp, touch)

    def show_box(self, axs=gp_Ax3(), lxyz=[100, 100, 100]):
        box = make_box(*lxyz)
        ax1 = gp_Ax3(
            gp_Pnt(-lxyz[0] / 2, -lxyz[1] / 2, -lxyz[2] / 2),
            gp_Dir(0, 0, 1)
        )
        trf = gp_Trsf()
        trf.SetTransformation(axs, gp_Ax3())
        trf.SetTransformation(ax1, gp_Ax3())
        box.Location(TopLoc_Location(trf))
        self.display.DisplayShape(axs.Location())
        self.show_axs_pln(axs, scale=lxyz[0])
        self.display.DisplayShape(box, transparency=0.7)

    def show_pnt(self, xyz=[0, 0, 0]):
        self.display.DisplayShape(gp_Pnt(*xyz))

    def show_pts(self, pts=[gp_Pnt()], num=1):
        for p in pts[::num]:
            self.display.DisplayShape(p)
        self.display.DisplayShape(make_polygon(pts))

    def show_ball(self, scale=100, trans=0.5):
        shape = BRepPrimAPI_MakeSphere(scale).Shape()
        self.display.DisplayShape(shape, transparency=trans)

    def show_vec(self, beam=gp_Ax3(), scale=1.0):
        pnt = beam.Location()
        vec = dir_to_vec(beam.Direction()).Scaled(scale)
        print(vec.Magnitude())
        self.display.DisplayVector(vec, pnt)

    def show_ellipsoid(self, axs=gp_Ax3(), rxyz=[10., 10., 10.], trans=0.5):
        shape = gen_ellipsoid(axs, rxyz)
        self.display.DisplayShape(shape, transparency=trans, color="BLUE1")
        return shape

    def show_axs_vec(self, beam=gp_Ax3(), scale=1.0):
        pnt = beam.Location()
        vec = dir_to_vec(beam.Direction()).Scaled(scale)
        self.display.DisplayVector(vec, pnt)

    def show_axs_pln(self, axs=gp_Ax3(), scale=100, name=None):
        pnt = axs.Location()
        dx = axs.XDirection()
        dy = axs.YDirection()
        dz = axs.Direction()
        vx = dir_to_vec(dx).Scaled(1 * scale)
        vy = dir_to_vec(dy).Scaled(1 * scale)
        vz = dir_to_vec(dz).Scaled(1 * scale)

        pnt_x = pnt_trf_vec(pnt, vx)
        pnt_y = pnt_trf_vec(pnt, vy)
        pnt_z = pnt_trf_vec(pnt, vz)
        lx, ly, lz = make_line(pnt, pnt_x), make_line(
            pnt, pnt_y), make_line(pnt, pnt_z)
        self.display.DisplayShape(pnt)
        self.display.DisplayShape(lx, color="RED")
        self.display.DisplayShape(ly, color="GREEN")
        self.display.DisplayShape(lz, color="BLUE1")
        if name != None:
            self.display.DisplayMessage(axs.Location(), name)
        return [lx, ly, lz]

    def show_plane(self, axs=gp_Ax3(), scale=100):
        pnt = axs.Location()
        vec = dir_to_vec(axs.Direction())
        pln = make_plane(pnt, vec, -scale, scale, -scale, scale)
        self.display.DisplayShape(pln)

    def show_wire(self, pts=[], axs=gp_Ax3()):
        poly = make_polygon(pts)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        # n_sided.Build()
        #face = n_sided.Face()
        self.display.DisplayShape(poly)
        return poly

    def prop_axs(self, axs=gp_Ax3(), scale=100, xyz="z"):
        if xyz == "x":
            vec = dir_to_vec(axs.XDirection()).Scaled(scale)
        elif xyz == "y":
            vec = dir_to_vec(axs.YDirection()).Scaled(scale)
        elif xyz == "z":
            vec = dir_to_vec(axs.Direction()).Scaled(scale)
        else:
            vec = dir_to_vec(axs.Direction()).Scaled(scale)
        return axs.Translated(vec)

    def make_plane_axs(self, axs=gp_Ax3(), rx=[0, 500], ry=[0, 500]):
        pln = BRepBuilderAPI_MakeFace(
            gp_Pln(axs),
            rx[0], rx[1], ry[0], ry[1]
        ).Face()
        return pln

    def make_circle(self, axs=gp_Ax3(), radi=100):
        return make_wire(make_edge(Geom_Circle(axs.Ax2(), radi)))

    def make_torus(self, axs=gp_Ax3(), r0=6000, r1=1500):
        tok_surf = Geom_ToroidalSurface(axs, r0, r1)
        return make_face(tok_surf, 1.0E-9)

    def make_cylinder_surf(self, axs=gp_Ax3(), radii=700, hight=500, rng=[0, 0.1], xyz="y"):
        loc = self.prop_axs(axs, radii, "z")
        if xyz == "y":
            rotate_xyz(loc, deg=90, xyz="y")
        elif xyz == "x":
            rotate_xyz(loc, deg=90, xyz="x")
            rotate_xyz(loc, deg=-90, xyz="z")
        else:
            loc = self.prop_axs(loc, -radii, "z")
            #loc = self.prop_axs(loc, -radii, "x")

        face = BRepBuilderAPI_MakeFace(
            gp_Cylinder(loc, radii),
            rng[0], rng[1],
            -hight / 2, hight / 2
        ).Face()
        return face

    def make_trimmedcylinder(self, axs=gp_Ax1(), radii=700, hight=500, rng=[0, 0.1], xyz="y"):
        loc = self.prop_axs(axs, radii, "z")
        if xyz == "y":
            rotate_xyz(loc, deg=90, xyz="y")
        elif xyz == "x":
            rotate_xyz(loc, deg=90, xyz="x")
            rotate_xyz(loc, deg=-90, xyz="z")
        else:
            loc = self.prop_axs(loc, -radii, "z")
            #loc = self.prop_axs(loc, -radii, "x")

        face = BRepBuilderAPI_MakeFace(
            gp_Cylinder(loc, radii),
            rng[0], rng[1],
            -hight / 2, hight / 2
        ).Face()
        return face

    def make_EllipWire(self, rxy=[1.0, 1.0], shft=0.0, axs=gp_Ax3()):
        rx, ry = rxy
        if rx > ry:
            major_radi = rx
            minor_radi = ry
            axis = gp_Ax2()
            axis.SetXDirection(axis.XDirection())
        else:
            major_radi = ry
            minor_radi = rx
            axis = gp_Ax2()
            axis.SetXDirection(axis.YDirection())
        axis.Rotate(axis.Axis(), np.deg2rad(shft))
        elip = make_edge(gp_Elips(axis, major_radi, minor_radi))
        poly = make_wire(elip)
        poly.Location(set_loc(gp_Ax3(), axs))
        return poly

    def make_PolyWire(self, num=6, radi=1.0, shft=0.0, axs=gp_Ax3(), skin=None):
        lxy = radi
        pnts = []
        angl = 360 / num
        for i in range(num):
            thet = np.deg2rad(i * angl) + np.deg2rad(shft)
            x, y = radi * np.sin(thet), radi * np.cos(thet)
            pnts.append(gp_Pnt(x, y, 0))
        pnts.append(pnts[0])
        poly = make_polygon(pnts)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        if skin == None:
            return poly
        elif skin == 0:
            return face
        else:
            solid = BRepOffset_MakeOffset(
                face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
            return solid.Shape()

    def make_StarWire(self, num=5, radi=[2.0, 1.0], shft=0.0, axs=gp_Ax3(), skin=None):
        lxy = radi
        pnts = []
        angl = 360 / num
        for i in range(num):
            a_thet = np.deg2rad(i * angl) + np.deg2rad(shft)
            ax, ay = radi[0] * np.sin(a_thet), radi[0] * np.cos(a_thet)
            pnts.append(gp_Pnt(ax, ay, 0))
            b_thet = a_thet + np.deg2rad(angl) / 2
            bx, by = radi[1] * np.sin(b_thet), radi[1] * np.cos(b_thet)
            pnts.append(gp_Pnt(bx, by, 0))
        pnts.append(pnts[0])
        poly = make_polygon(pnts)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        if skin == None:
            return poly
        elif skin == 0:
            return face
        else:
            solid = BRepOffset_MakeOffset(
                face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
            return solid.Shape()

    def make_Wire_pts(self, dat=[], axs=gp_Ax3()):
        num = dat.shape
        pts = []
        if num[1] == 2:
            for p in dat:
                pts.append(gp_Pnt(p[0], p[1], 0))
        elif num[1] == 3:
            for p in dat:
                pts.append(gp_Pnt(p[0], p[1], p[2]))
        else:
            for p in dat:
                pts.append(gp_Pnt(p[0], p[1], p[2]))
        pts = np.array(pts)
        #cov = ConvexHull(pts, qhull_options='QJ')

        #pts_ord = []
        # print(cov)
        # print(cov.simplices)
        # print(cov.vertices)
        # for idx in cov.vertices:
        #    print(idx, pnt[idx])
        #    pts_ord.append(gp_Pnt(*pnt[idx]))

        #poly = make_polygon(pts_ord)
        poly = make_polygon(pts)
        poly.Location(set_loc(gp_Ax3(), axs))
        #n_sided = BRepFill_Filling()
        # for e in Topo(poly).edges():
        #    n_sided.Add(e, GeomAbs_C0)
        # n_sided.Build()
        #face = n_sided.Face()
        return poly

    def make_skin(self, pts=[], axs=gp_Ax3(), skin=1.0):
        poly = make_polygon(pts, closed=True)
        poly.Location(set_loc(gp_Ax3(), axs))

        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        solid = BRepOffset_MakeOffset(
            face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
        return solid.Shape()

    def make_skin_wire(self, poly, axs=gp_Ax3(), skin=1.0):
        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        if skin == None:
            return poly
        elif skin == 0:
            return face
        else:
            solid = BRepOffset_MakeOffset(
                face, skin, 1.0E-5, BRepOffset_Skin, False, True, GeomAbs_Arc, True, True)
            return solid.Shape()

    def make_FaceByOrder(self, pts=[]):
        dat = []
        for p in pts:
            dat.append([p.X(), p.Y(), p.Z()])

        dat = np.array(dat)
        cov = ConvexHull(dat, qhull_options='QJ')

        #pts_ord = []
        # print(cov)
        # print(cov.simplices)
        # print(cov.vertices)
        # for idx in cov.vertices:
        #    print(idx, pnt[idx])
        #    pts_ord.append(gp_Pnt(*pnt[idx]))

        #poly = make_polygon(pts_ord)
        poly = make_polygon(pts)
        n_sided = BRepFill_Filling()
        for e in Topo(poly).edges():
            n_sided.Add(e, GeomAbs_C0)
        n_sided.Build()
        face = n_sided.Face()
        return dat, face

    def proj_rim_pln(self, wire, surf, axs=gp_Ax3()):
        proj = BRepProj_Projection(wire, surf, axs.Direction())
        return proj.Current()

    def proj_pnt_pln(self, pnt, surf, axs=gp_Ax3()):
        lin = gp_Lin(pnt, axs.Direction())
        api = GeomAPI_IntCS(Geom_Line(lin), BRep_Tool.Surface(surf))
        dst = np.inf
        sxy = pnt
        num = api.NbPoints()
        for i in range(num):
            p0 = api.Point(i + 1)
            dst0 = pnt.Distance(p0)
            if dst0 < dst:
                dst = dst0
                sxy = p0
        return sxy

    def proj_pln_show(self, face, nxy=[10, 10], ux=[0, 1], uy=[0, 1], axs=gp_Ax3()):
        trf = set_trf(gp_Ax3(), axs)
        pln = self.make_plane_axs(axs, [-1000, 1000], [-1000, 1000])
        surf = BRep_Tool.Surface(face)
        for px in np.linspace(ux[0], ux[1], nxy[0]):
            for py in np.linspace(uy[0], uy[1], nxy[1]):
                p0 = surf.Value(px, py)
                p1 = self.proj_pnt_pln(p0, pln, axs)
                self.display.DisplayShape(p0)
                self.display.DisplayShape(p1)

    def proj_pln_showup(self, face, nxy=[10, 10], lx=[-10, 10], ly=[-10, 10], axs=gp_Ax3()):
        trf = set_trf(gp_Ax3(), axs)
        nx, ny = nxy
        xs, xe = lx
        ys, ye = ly
        plnx = np.linspace(xs, xe, nx)
        plny = np.linspace(ys, ye, ny)
        mesh = np.meshgrid(plnx, plny)
        data = np.zeros_like(mesh[0])
        for (ix, iy), x in np.ndenumerate(data):
            px, py = mesh[0][ix, iy], mesh[1][ix, iy]
            p0 = gp_Pnt(px, py, 0).Transformed(trf)
            p1 = self.proj_pnt_pln(p0, face, axs)
            z = p0.Distance(p1)
            data[ix, iy] = z
            self.display.DisplayShape(p0)
            self.display.DisplayShape(p1)
        return mesh, data

    def reflect_beam(self, shpe=TopoDS_Shape(), beam0=gp_Ax3(), tr=0):
        v0 = dir_to_vec(beam0.Direction())
        v1 = dir_to_vec(beam0.XDirection())
        p0 = beam0.Location()
        lin = gp_Lin(beam0.Axis())
        api = BRepIntCurveSurface_Inter()
        api.Init(shpe, lin, 1.0E-9)
        dst = np.inf
        num = 0
        sxy = p0
        uvw = [0, 0, 0]
        fce = None
        while api.More():
            p1 = api.Pnt()
            dst1 = p0.Distance(p1)
            if dst1 < dst and api.W() > 1.0E-6:
                dst = dst1
                uvw = [api.U(), api.V(), api.W()]
                sxy = api.Pnt()
                fce = api.Face()
                api.Next()
            else:
                api.Next()

        print(*uvw)
        u, v, w = uvw
        surf = BRepAdaptor_Surface(fce)
        prop = BRepLProp_SLProps(surf, u, v, 2, 1.0E-9)
        p1, vx, vy = prop.Value(), prop.D1U(), prop.D1V()
        vz = vx.Crossed(vy)
        if vz.Dot(v0) > 0:
            vz.Reverse()
        vx.Normalize()
        vy.Normalize()
        vz.Normalize()
        beam1 = gp_Ax3(
            p1,
            vec_to_dir(v0.Reversed()),
            vec_to_dir(v1.Reversed())
        )
        norm1 = gp_Ax3(
            p1,
            vec_to_dir(vz),
            vec_to_dir(vx)
        )
        if tr == 0:
            # Reflect
            beam1.Mirror(norm1.Ax2())
            if beam1.Direction().Dot(norm1.Direction()) < 0:
                beam1.ZReverse()
        elif tr == 1:
            # Transporse
            beam1.ZReverse()
            beam1.XReverse()
        return beam1

    def calc_angle(self, ax0=gp_Ax3(), ax1=gp_Ax3()):
        p0 = ax0.Location()
        v0 = dir_to_vec(ax0.Direction())
        p1 = ax1.Location()
        v1 = dir_to_vec(ax1.Direction())
        return p0.Distance(p1), v0.Dot(v1)

    def calc_tor_angle1(self, ax0=gp_Ax3(), ax1=gp_Ax3()):
        # img/tok_angle_def1.png
        print(dir_to_vec(ax1.Direction()))
        # Base plane on ax0
        pln = self.make_plane_axs(ax0, [-1, 1], [-1, 1])

        # Beam Project to Base plane
        ax1_prj = self.proj_pnt_pln(ax1.Location(), pln, ax0)
        ax1_dst = self.prop_axs(ax1, scale=1).Location()

        # Ref Coord
        # Z Direction: ax1-pnt -> ax0-Z-Ax
        # Y Direction: ax0-ZDir
        ax1_loc = gp_Ax3(ax1.Ax2())
        ax1_loc.SetDirection(vec_to_dir(gp_Vec(ax1_prj, ax0.Location())))
        ax1_loc.SetYDirection(ax0.Direction())
        v_z = dir_to_vec(ax1_loc.Direction())
        v_x = dir_to_vec(ax1_loc.XDirection())
        v_y = dir_to_vec(ax1_loc.YDirection())

        # Ref Coord XY-Plane (Poloidal)
        ax1_axy = gp_Ax3(ax1.Location(),
                         ax1_loc.Direction(),
                         ax1_loc.XDirection())
        ax1_pxy = self.make_plane_axs(ax1_axy, [-1, 1], [-1, 1])
        pnt_pxy = self.proj_pnt_pln(ax1_dst, ax1_pxy, ax1_loc)

        # Ref Coord XZ-Plane (Toroidal)
        ax1_axz = gp_Ax3(ax1.Location(),
                         ax1_loc.YDirection(),
                         ax1_loc.XDirection())
        ax1_pxz = self.make_plane_axs(ax1_axz, [-1, 1], [-1, 1])
        pnt_pxz = self.proj_pnt_pln(ax1_dst, ax1_pxz, ax1_axz)
        lin_pxz = make_edge(ax1_loc.Location(), pnt_pxz)
        vec_pxz = gp_Vec(ax1_loc.Location(), pnt_pxz)
        deg_pxz = v_z.AngleWithRef(vec_pxz, v_y.Reversed())
        txt_pxz = "pnt_pxy: {:.1f}".format(np.rad2deg(deg_pxz))
        print("Tor: ", np.rad2deg(deg_pxz))

        # Ref Coord YZ-Plane (Poloidal)
        ax1_ayz = gp_Ax3(ax1.Location(),
                         ax1_loc.XDirection(),
                         ax1_loc.Direction())
        ax1_pyz = self.make_plane_axs(ax1_ayz, [-1, 1], [-1, 1])
        pnt_pyz = self.proj_pnt_pln(ax1_dst, ax1_pyz, ax1_ayz)
        lin_pyz = make_edge(ax1_loc.Location(), pnt_pyz)
        vec_pyz = gp_Vec(ax1_loc.Location(), pnt_pyz)
        deg_pyz = np.pi / 2 - v_y.AngleWithRef(vec_pyz, v_x)
        txt_pyz = "pnt_pyz: {:.1f}".format(np.rad2deg(deg_pyz))
        print("Pol: ", np.rad2deg(deg_pyz))

        #self.show_axs_pln(ax0, scale=0.5)
        #self.show_axs_pln(ax1_loc, scale=1)
        #self.display.DisplayShape(lin_pxz, color="BLUE1")
        #self.display.DisplayShape(ax1_pxz, color="BLUE1", transparency=0.9)
        #self.display.DisplayShape(lin_pyz, color="GREEN")
        #self.display.DisplayShape(ax1_pyz, color="GREEN", transparency=0.9)
        # self.display.DisplayShape(
        #   make_edge(ax1.Location(), ax1_dst), color="YELLOW")
        # self.display.DisplayVector(dir_to_vec(
        #   ax1.Direction()).Scaled(0.5), ax1_dst)
        return deg_pxz, deg_pyz

    def calc_tor_angle2(self, ax0=gp_Ax3(), ax1=gp_Ax3()):
        # img/tok_angle_def2.png
        print(dir_to_vec(ax1.Direction()))
        # Base plane on ax0
        pln = self.make_plane_axs(ax0, [-1, 1], [-1, 1])

        # Beam Project to Base plane
        ax1_prj = self.proj_pnt_pln(ax1.Location(), pln, ax0)
        ax1_dst = self.prop_axs(ax1, scale=1).Location()
        ax1_dst_prj = self.proj_pnt_pln(ax1_dst, pln, ax0)

        # Ref Coord
        # Z Direction: ax1-dir (projected)
        # Y Direction: ax0-ZDir
        ax1_loc = gp_Ax3(ax1.Ax2())
        ax1_loc.SetDirection(vec_to_dir(gp_Vec(ax1_prj, ax1_dst_prj)))
        ax1_loc.SetYDirection(ax0.Direction())
        v_z = dir_to_vec(ax1_loc.Direction())
        v_x = dir_to_vec(ax1_loc.XDirection())
        v_y = dir_to_vec(ax1_loc.YDirection())

        # Ref Coord XY-Plane (Poloidal)
        ax1_axy = gp_Ax3(ax1.Location(),
                         ax1_loc.Direction(),
                         ax1_loc.XDirection())
        ax1_pxy = self.make_plane_axs(ax1_axy, [-1, 1], [-1, 1])
        pnt_pxy = self.proj_pnt_pln(ax1_dst, ax1_pxy, ax1_loc)

        # Ref Coord XZ-Plane (Toroidal)
        ax1_axz = gp_Ax3(ax1.Location(),
                         ax1_loc.YDirection(),
                         ax1_loc.XDirection())
        ax1_pxz = self.make_plane_axs(ax1_axz, [-1, 1], [-1, 1])
        pnt_pxz = self.proj_pnt_pln(ax1_dst, ax1_pxz, ax1_axz)
        lin_pxz = make_edge(ax1_loc.Location(), pnt_pxz)
        vec_pxz = gp_Vec(ax1_prj, ax0.Location())
        deg_pxz = v_z.AngleWithRef(vec_pxz, v_y)
        txt_pxz = "pnt_pxy: {:.1f}".format(np.rad2deg(deg_pxz))
        print("Tor: ", np.rad2deg(deg_pxz))

        # Ref Coord YZ-Plane (Poloidal)
        ax1_ayz = gp_Ax3(ax1.Location(),
                         ax1_loc.XDirection(),
                         ax1_loc.Direction())
        ax1_pyz = self.make_plane_axs(ax1_ayz, [-1, 1], [-1, 1])
        pnt_pyz = self.proj_pnt_pln(ax1_dst, ax1_pyz, ax1_ayz)
        lin_pyz = make_edge(ax1_loc.Location(), pnt_pyz)
        vec_pyz = dir_to_vec(ax1.Direction())
        deg_pyz = vec_pyz.AngleWithRef(v_z, v_x)
        txt_pyz = "pnt_pyz: {:.1f}".format(np.rad2deg(deg_pyz))
        print("Pol: ", np.rad2deg(deg_pyz))

        #self.show_axs_pln(ax0, scale=0.5)
        #self.show_axs_pln(ax1_loc, scale=1)
        #self.display.DisplayShape(lin_pxz, color="BLUE1")
        #self.display.DisplayShape(ax1_pxz, color="BLUE1", transparency=0.9)
        #self.display.DisplayShape(lin_pyz, color="GREEN")
        #self.display.DisplayShape(ax1_pyz, color="GREEN", transparency=0.9)
        # self.display.DisplayShape(
        #   make_edge(ax1.Location(), ax1_dst), color="YELLOW")
        # self.display.DisplayVector(dir_to_vec(
        #   ax1.Direction()).Scaled(0.5), ax1_dst)
        return deg_pxz, deg_pyz


class Face (object):

    def __init__(self, axs=gp_Ax3(), *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.axis = axs
        self.face = make_plane()
        self.MoveSurface(ax2=self.axis)

    def MoveSurface(self, ax1=gp_Ax3(), ax2=gp_Ax3()):
        trsf = set_trf(ax1, ax2)
        self.face.Move(TopLoc_Location(trsf))

    def rot_axs(self, pxyz=[0, 0, 0], rxyz=[0, 0, 0]):
        axs = gp_Ax3(gp_Pnt(*pxyz), gp_Dir(0, 0, 1))
        ax1 = gp_Ax1(self.axis.Location(), self.axis.XDirection())
        ax2 = gp_Ax1(self.axis.Location(), self.axis.YDirection())
        ax3 = gp_Ax1(self.axis.Location(), self.axis.Direction())
        axs.Rotate(ax1, np.deg2rad(rxyz[0]))
        axs.Rotate(ax2, np.deg2rad(rxyz[1]))
        axs.Rotate(ax3, np.deg2rad(rxyz[2]))
        trsf = set_trf(gp_Ax3(), axs)
        self.axis.Transform(trsf)
        self.face.Move(TopLoc_Location(trsf))


class OCCSurfObj(object):

    def __init__(self, name="surf"):
        self.axs = gp_Ax3()
        self.rot = self.axs
        self.rim = make_edge(gp_Circ(self.axs.Ax2(), 100))
        self.pln = plotocc.make_plane_axs(self.axs)
        self.surf = make_plane(
            self.axs.Location(), dir_to_vec(self.axs.Direction()),
            -500, 500, -500, 500)
        self.face = self.BuilFace()
        self.name = name

    def get_trsf(self):
        self.trf = gp_Trsf()
        self.trf.SetTransformation(self.axs, gp_Ax3())
        return self.trf

    def get_vxyz(self):
        trf = gp_Trsf()
        trf.SetTransformation(gp_Ax3(), self.rot)
        # trf.Invert()
        ax1 = self.axs.Transformed(trf)
        d_z = ax1.Direction()
        print(dir_to_vec(d_z),
              get_deg(self.axs, dir_to_vec(self.rot.Direction())))
        return [d_z.X(), d_z.Y(), d_z.Z()]

    def BuilFace(self):
        proj = BRepProj_Projection(self.rim, self.surf, self.axs.Direction())
        bild = BRepBuilderAPI_MakeFace(self.surf, proj.Current())
        return bild.Face()

    def UpdateFace(self):
        self.face = self.BuilFace()

    def RotateFace(self, deg=0.0, axs="z"):
        if axs == "x":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.XDirection())
        elif axs == "y":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.YDirection())
        elif axs == "z":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        else:
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        rot = self.rot.Rotated(ax1, np.deg2rad(deg))
        trf = gp_Trsf()
        trf.SetDisplacement(self.rot, rot)
        self.axs.Transform(trf)
        self.face.Move(TopLoc_Location(trf))

    def RotateSurf(self, deg=0.0, axs="z"):
        if axs == "x":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.XDirection())
        elif axs == "y":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.YDirection())
        elif axs == "z":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        else:
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        trf = gp_Trsf()
        trf.SetRotation(ax1, np.deg2rad(deg))
        self.rot.Transform(trf)
        self.axs.Transform(trf)
        self.surf.Move(TopLoc_Location(trf))

    def RotateSurf2(self, deg=0.0, axs="z"):
        if axs == "x":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.XDirection())
        elif axs == "y":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.YDirection())
        elif axs == "z":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        else:
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        trf = gp_Trsf()
        trf.SetRotation(ax1, np.deg2rad(deg))
        self.axs.Transform(trf)
        self.surf.Move(TopLoc_Location(trf))

    def SetSurf_XY(self, deg=0.0, axs="x"):
        dx, dy = get_deg(self.axs, dir_to_vec(self.rot.Direction()))
        if axs == "x":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.XDirection())
            dg0 = np.rad2deg(dy)
        elif axs == "y":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.YDirection())
            dg0 = np.rad2deg(dx)
        else:
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
            dg0 = np.rad2deg(dx)
        trf = gp_Trsf()
        trf.SetRotation(ax1, np.deg2rad(-dg0 + deg))
        self.axs.Transform(trf)
        self.surf.Move(TopLoc_Location(trf))
        get_deg(self.rot, dir_to_vec(self.axs.Direction()))

    def MovXYZSurf(self, dst=0.0, axs="z"):
        if axs == "x":
            vec = dir_to_vec(self.rot.XDirection())
        elif axs == "y":
            vec = dir_to_vec(self.rot.YDirection())
        elif axs == "z":
            vec = dir_to_vec(self.rot.Direction())
        else:
            vec = dir_to_vec(self.rot.Direction())
        trf = gp_Trsf()
        trf.SetTranslation(vec.Scaled(dst))
        # self.rot.Transform(trf)
        self.axs.Transform(trf)
        self.surf.Move(TopLoc_Location(trf))

    def MovVecSurf(self, vec=gp_Vec(0, 0, 1)):
        trf = gp_Trsf()
        trf.SetTranslation(vec)
        self.rot.Transform(trf)
        self.axs.Transform(trf)
        self.surf.Move(TopLoc_Location(trf))

    def RotateAxis(self, deg=0.0, axs="z"):
        if axs == "x":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.XDirection())
        elif axs == "y":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.YDirection())
        elif axs == "z":
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        else:
            ax1 = gp_Ax1(self.rot.Location(), self.rot.Direction())
        rot = self.rot.Rotated(ax1, np.deg2rad(deg))
        trf = gp_Trsf()
        trf.SetDisplacement(self.rot, rot)
        self.axs.Transform(trf)

    def RotateAxis_Ax(self, ax=gp_Ax3(), deg=0.0, axs="z"):
        if axs == "x":
            ax1 = gp_Ax1(ax.Location(), ax.XDirection())
        elif axs == "y":
            ax1 = gp_Ax1(ax.Location(), ax.YDirection())
        elif axs == "z":
            ax1 = gp_Ax1(ax.Location(), ax.Direction())
        else:
            ax1 = gp_Ax1(ax.Location(), ax.Direction())
        rot = ax.Rotated(ax1, np.deg2rad(deg))
        trf = gp_Trsf()
        trf.SetDisplacement(ax, rot)
        ax.Transform(trf)

    def MoveRel(self, trf=gp_Trsf()):
        self.axs.Transform(trf)
        self.face.Move(TopLoc_Location(trf))

    def SurfCurvature(self, nxy=[200, 200], lxy=[450, 450], rxy=[700, 0], sxy=[0, 0]):
        px = np.linspace(-1, 1, int(nxy[0])) * lxy[0] / 2
        py = np.linspace(-1, 1, int(nxy[1])) * lxy[1] / 2
        mesh = np.meshgrid(px, py)
        surf_x = curvature(mesh[0], r=rxy[0], s=sxy[0])
        surf_y = curvature(mesh[1], r=rxy[1], s=sxy[1])
        data = surf_x + surf_y
        self.surf, self.surf_pts = surf_spl_pcd(*mesh, data)

    def SurfCurvature_Loc(self, nxy=[200, 200], lxy=[450, 450], rxy=[700, 0], sxy=[0, 0]):
        px = np.linspace(-1, 1, int(nxy[0])) * lxy[0] / 2
        py = np.linspace(-1, 1, int(nxy[1])) * lxy[1] / 2
        mesh = np.meshgrid(px, py)
        surf_x = curvature(mesh[0], r=rxy[0], s=sxy[0])
        surf_y = curvature(mesh[1], r=rxy[1], s=sxy[1])
        data = surf_x + surf_y
        self.surf, self.surf_pts = surf_spl_pcd(*mesh, data)
        trf = gp_Trsf()
        trf.SetTransformation(self.axs, gp_Ax3())
        self.surf.Location(TopLoc_Location(trf))

    def get_surf_uvpnt(self, uv=[0, 0]):
        surf = BRep_Tool.Surface(self.surf)
        pnt = gp_Pnt()
        surf.D0(uv[0], uv[1], pnt)
        return pnt

    def load_rim(self, rimfile="../ticra/input/surf/mou.rim"):
        data = np.loadtxt(rimfile, skiprows=2)
        pts = []
        for xy in data:
            pts.append(gp_Pnt(*xy, 0))
        self.rim = make_polygon(pts, closed=True)

    def load_mat(self, sfcfile="../ticra/input/surf/pln_mat.sfc"):
        xs, ys, xe, ye = [float(v) for v in getline(sfcfile, 2).split()]
        nx, ny = [int(v) for v in getline(sfcfile, 3).split()]
        px = np.linspace(xs, xe, nx)
        py = np.linspace(ys, ye, ny)
        mesh = np.meshgrid(px, py)
        data = np.loadtxt(sfcfile, skiprows=3).T
        self.surf, self.surf_pts = surf_spl_pcd(*mesh, data)

    def export_rim_2d(self, rimfile="m2.rim", name="m2-rim"):
        rim_2d = plotocc.proj_rim_pln(self, self.rim, self.pln, self.axs)

        fp = open(rimfile, "w")
        fp.write(' {:s}\n'.format(name))
        fp.write('{:12d}{:12d}{:12d}\n'.format(1, 1, 1))
        rim_tmp = gp_Pnt()
        for i, e in enumerate(Topo(rim_2d).edges()):
            e_curve, u0, u1 = BRep_Tool.Curve(e)
            print(i, e, u0, u1)
            if i != 0 and rim_tmp == e_curve.Value(u0):
                u_range = np.linspace(u0, u1, 50)
                rim_tmp = e_curve.Value(u1)
                p = e_curve.Value(u0)
                p.Transform(set_trf(self.axs, gp_Ax3()))
                data = [p.X(), p.Y()]
                print(0, p, u_range[0], u_range[-1])
            elif i != 0 and rim_tmp == e_curve.Value(u1):
                u_range = np.linspace(u1, u0, 50)
                rim_tmp = e_curve.Value(u0)
                p = e_curve.Value(u1)
                p.Transform(set_trf(self.axs, gp_Ax3()))
                data = [p.X(), p.Y()]
                print(1, p, u_range[0], u_range[-1])
            else:
                u_range = np.linspace(u0, u1, 50)
                rim_tmp = e_curve.Value(u1)
                p = e_curve.Value(u0)
                p.Transform(set_trf(self.axs, gp_Ax3()))
                data = [p.X(), p.Y()]
                print(2, p, u_range[0], u_range[-1])
            fp.write(''.join([float_to_string(val) for val in data]) + '\n')
            for u in u_range[1:]:
                p = e_curve.Value(u)
                p.Transform(set_trf(self.axs, gp_Ax3()))
                data = [p.X(), p.Y()]
                fp.write(''.join([float_to_string(val)
                                  for val in data]) + '\n')
        fp.close()

    def export_sfc1_axs(self, sfcfile="m2_mat.sfc", name="M2 Mat"):
        surf = BRep_Tool.Surface(self.surf)

        trf = set_trf(self.axs, gp_Ax3())
        xy0 = plotocc.proj_pnt_pln(self, surf.Value(0, 0), self.pln, self.axs)
        xy1 = plotocc.proj_pnt_pln(self, surf.Value(1, 1), self.pln, self.axs)
        xy0.Transform(trf)
        xy1.Transform(trf)

        m2_trf = set_trf(gp_Ax3(), self.axs)
        m2_pln = BRep_Tool.Surface(self.pln)
        for px in np.linspace(-100, 100, 10):
            for py in np.linspace(-100, 100, 10):
                p0 = gp_Pnt(px, py, 0).Transformed(m2_trf)
                p1 = plotocc.proj_pnt_pln(None, p0, self.surf, self.axs)

        #ix0, ix1 = m2.surf_pts.LowerRow(), m2.surf_pts.UpperRow()
        #iy0, iy1 = m2.surf_pts.LowerCol(), m2.surf_pts.UpperCol()
        #xy0 = m2.surf_pts.Value(ix0, iy0).Transformed(trf)
        #xy1 = m2.surf_pts.Value(ix1, iy1).Transformed(trf)
        nx, ny = 200, 200
        xs, xe = xy0.X(), xy1.X()
        ys, ye = xy0.Y(), xy1.Y()
        fp = open(sfcfile, "w")
        fp.write(" {} \n".format(name))
        fp.write(" {:.2e} {:.2e} {:.2e} {:.2e}\n".format(xs, ys, xe, ye))
        fp.write(" {:d} {:d}\n".format(nx, ny))
        for ix in np.linspace(0, 1, nx):
            for iy in np.linspace(0, 1, ny):
                p0 = surf.Value(ix, iy)
                p1 = plotocc.proj_pnt_pln(self, p0, self.pln, self.axs)
                pz = p1.Transformed(trf)
                z = p0.Distance(p1)
                fp.write(" {:.5e} ".format(z))
            fp.write("\n")
        fp.close()
        print(xy0)

    def export_sfc2_axs(self, nxy=[200, 200], rx=[-250, 250], ry=[-250, 250], sfcfile="m2_mat.sfc"):
        trsf = set_trf(gp_Ax3(), self.axs)
        nx, ny = nxy
        xs, xe = rx
        ys, ye = ry
        plnx = np.linspace(xs, xe, nx)
        plny = np.linspace(ys, ye, ny)
        mesh = np.meshgrid(plnx, plny)
        data = np.zeros_like(mesh[0])
        for (ix, iy), x in np.ndenumerate(data):
            px, py = mesh[0][ix, iy], mesh[1][ix, iy]
            p0 = gp_Pnt(px, py, 0).Transformed(trsf)
            p1 = plotocc.proj_pnt_pln(self, p0, self.surf, self.axs)
            z = p0.Distance(p1)
            data[ix, iy] = z

            txt = "\r"
            txt += "{:d}, {:d} / {:d}, {:d}".format(ix, iy, ny, nx)
            sys.stdout.write(txt)
            sys.stdout.flush()
        print()
        grasp_sfc(mesh, data, sfcfile)

    def reflect_beam(self, beam0=gp_Ax3(), tr=0):
        v0 = dir_to_vec(beam0.Direction())
        v1 = dir_to_vec(beam0.XDirection())
        p0 = beam0.Location()
        lin = gp_Lin(beam0.Axis())
        api = BRepIntCurveSurface_Inter()
        api.Init(self.surf, lin, 1.0E-9)
        dst = np.inf
        num = 0
        sxy = p0
        uvw = [0, 0, 0]
        fce = None
        while api.More():
            p1 = api.Pnt()
            dst1 = p0.Distance(p1)
            if dst1 < dst and api.W() > 1.0E-6:
                dst = dst1
                uvw = [api.U(), api.V(), api.W()]
                sxy = api.Pnt()
                fce = api.Face()
                api.Next()
            else:
                api.Next()
        if fce == None:
            fce = make_plane(self.axs.Location(), dir_to_vec(self.axs.Direction()),
                             -10000, 10000, -10000, 10000)
            api.Init(fce, lin, 1.0E-9)
            uvw = [api.U(), api.V(), api.W()]

        print(self.name, *uvw)
        u, v, w = uvw
        surf = BRepAdaptor_Surface(fce)
        prop = BRepLProp_SLProps(surf, u, v, 2, 1.0E-9)
        p1, vx, vy = prop.Value(), prop.D1U(), prop.D1V()
        vz = vx.Crossed(vy)
        if vz.Dot(v0) > 0:
            vz.Reverse()
        vx.Normalize()
        vy.Normalize()
        vz.Normalize()

        self.beam = gp_Ax3(
            p1,
            vec_to_dir(v0.Reversed()),
            vec_to_dir(v1.Reversed())
        )
        self.norm = gp_Ax3(
            p1,
            vec_to_dir(vz),
            vec_to_dir(vx)
        )
        if tr == 0:
            # Reflect
            self.beam.Mirror(self.norm.Ax2())
            if self.beam.Direction().Dot(self.norm.Direction()) < 0:
                self.beam.ZReverse()
        elif tr == 1:
            # Transporse
            self.beam.ZReverse()
            self.beam.XReverse()
            # if self.beam.Direction().Dot(self.norm.Direction()) < 0:
            #    self.beam.ZReverse()
        # print(self.beam.Direction().Dot(self.norm.Direction()))


if __name__ == '__main__':
    tmpdir = create_tempdir(flag=-1)
    print(tmpdir)

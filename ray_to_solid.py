import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.constants as cnt
from optparse import OptionParser

from OCC.Coregp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Coregp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.CoreBRep import BRep_Tool
from OCC.CoreBRepTools import BRepTools_GTrsfModification
from OCC.CoreBRepCheck import BRepCheck_Shell
from OCC.CoreBRepBuilderAPI import BRepBuilderAPI_Collect, BRepBuilderAPI_LineThroughIdenticPoints
from OCC.CoreBRepAlgoAPI import BRepAlgoAPI_Common
from OCC.CoreBRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.CoreGeom import Geom_Line
from OCC.CoreGeomAPI import GeomAPI_IntCS
from OCC.CoreGeomLProp import GeomLProp_SurfaceTool
from OCC.CoreShapeAlgo import ShapeAlgo_ToolContainer
from OCC.CoreIntAna import IntAna_Line
from OCC.CoreIntCurveSurface import IntCurveSurface_In
from OCC.CoreIntPatch import IntPatch_TheSearchInside
from OCCUtils.Construct import vec_to_dir, dir_to_vec
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import make_plane, make_polygon, make_box

from base import plotocc, Face, set_trf, rot_axs, set_loc
from Surface import surf_curv


def move_pnt_to_dir(axs=gp_Ax3(), scale=100):
    vec = point_to_vector(axs.Location()) + dir_to_vec(axs.Direction()) * scale
    return vector_to_point(vec)


class TraceSystem (plotocc):

    def __init__(self):
        plotocc.__init__(self)
        self.axis = gp_Ax3(gp_Pnt(1000, 1000, 1000), gp_Dir(0, 0, 1))
        self.trsf = set_trf(ax2=self.axis)
        self.box = make_box(gp_Pnt(), 250, 250, 250)
        self.box.Location(set_loc(ax2=self.axis))

        self.beam1 = gp_Ax3()
        rot_axs(self.beam1, [100, 100, -500], [0, 0, 0])
        self.beam1.Transform(self.trsf)

        self.Reflect(self.beam1, self.box)

        self.beam2 = gp_Ax3()
        rot_axs(self.beam2, [0, 0, -500], [0, 1, 0])
        self.beam2.Transform(self.trsf)

        self.beam3 = gp_Ax3()
        rot_axs(self.beam3, [-100, -100, -500], [0, 2, 0])
        self.beam3.Transform(self.trsf)

    def run_beam(self, beam=gp_Ax3()):
        pts = [beam.Location()]

        print(beam.Location(), dir_to_vec(beam.Direction()))
        beam = self.Reflect(beam, self.surf1.face)
        pts.append(beam.Location())

        print(beam.Location(), dir_to_vec(beam.Direction()))
        beam = self.Reflect(beam, self.surf2.face)
        pts.append(beam.Location())

        pnt = move_pnt_to_dir(beam, 2500)
        pts.append(pnt)

        beam_ray = make_polygon(pts)
        return beam, beam_ray

    def Reflect(self, beam=gp_Ax3(), surf=make_plane()):
        h_surf = BRep_Tool.Surface(surf)
        ray = Geom_Line(beam.Location(), beam.Direction()).GetHandle()
        if GeomAPI_IntCS(ray, h_surf).NbPoints() == 0:
            beam_v1 = beam
        else:
            GeomAPI_IntCS(ray, h_surf).IsDone()
            uvw = GeomAPI_IntCS(ray, h_surf).Parameters(1)
            u, v, w = uvw
            p1, vx, vy = gp_Pnt(), gp_Vec(), gp_Vec()
            GeomLProp_SurfaceTool.D1(h_surf, u, v, p1, vx, vy)
            vz = vx.Crossed(vy)
            vx.Normalize()
            vy.Normalize()
            vz.Normalize()
            norm = gp_Ax3(p1, vec_to_dir(vz), vec_to_dir(vx))
            beam_v0 = beam
            beam_v0.SetLocation(p1)
            beam_v1 = beam_v0.Mirrored(norm.Ax2())
            beam_v1.XReverse()
        return beam_v1

    def Display(self):
        self.display.DisplayShape(self.box)

        self.show_axs_pln(self.axis, scale=10)
        self.show_axs_pln(self.beam1, scale=10)


if __name__ == "__main__":
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./")
    opt, argc = parser.parse_args(argvs)
    print(argc, opt)

    obj = TraceSystem()
    obj.Display()
    obj.show()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.constants as cnt
from optparse import OptionParser

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_Line
from OCC.Core.GeomAPI import GeomAPI_IntCS
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool
from OCCUtils.Construct import vec_to_dir, dir_to_vec
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import make_plane, make_polygon

from base import plotocc, Face, set_trf, rot_axs
from Surface import surf_curv


def move_pnt_to_dir(axs=gp_Ax3(), scale=100):
    vec = point_to_vector(axs.Location()) + dir_to_vec(axs.Direction()) * scale
    return vector_to_point(vec)


class TraceSystem (plotocc):

    def __init__(self):
        plotocc.__init__(self)
        self.axis = gp_Ax3(gp_Pnt(1000, 1000, 1000), gp_Dir(0, 0, 1))
        self.trsf = set_trf(ax2=self.axis)

        ax1 = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, -1))
        ax1.Transform(self.trsf)
        self.surf1 = Face(ax1)
        self.surf1.face = surf_curv(lxy=[400, 300], rxy=[3000, 3000])
        self.surf1.MoveSurface(ax2=self.surf1.axis)
        self.surf1.rot_axs(rxyz=[0, 50, 0])

        ax2 = gp_Ax3(gp_Pnt(-1000, 0, 200), gp_Dir(0, 0, 1))
        ax2.Transform(self.trsf)
        self.surf2 = Face(ax2)
        self.surf2.face = surf_curv(lxy=[300, 200], rxy=[3000, -3000])
        self.surf2.MoveSurface(ax2=self.surf2.axis)
        self.surf2.rot_axs(pxyz=[0, -50, 10], rxyz=[20, 60, 0])

        ax3 = gp_Ax3(gp_Pnt(-750, 0, 1500), gp_Dir(0, 0, 1))
        ax3.Transform(self.trsf)
        self.surf3 = Face(ax3)
        self.surf3.face = surf_curv(lxy=[2500, 2500], rxy=[0, 0])
        self.surf3.MoveSurface(ax2=self.surf3.axis)

        self.beam1 = gp_Ax3()
        rot_axs(self.beam1, [100, 100, -500], [0, 0, 0])
        self.beam1.Transform(self.trsf)
        self.beam1, self.ray1 = self.run_beam(self.beam1)

        self.beam2 = gp_Ax3()
        rot_axs(self.beam2, [0, 0, -500], [0, 1, 0])
        self.beam2.Transform(self.trsf)
        self.beam2, self.ray2 = self.run_beam(self.beam2)

        self.beam3 = gp_Ax3()
        rot_axs(self.beam3, [-100, -100, -500], [0, 2, 0])
        self.beam3.Transform(self.trsf)
        self.beam3, self.ray3 = self.run_beam(self.beam3)

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
        ray = Geom_Line(beam.Location(), beam.Direction())
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
        self.display.DisplayShape(self.surf1.face)
        self.display.DisplayShape(self.surf2.face)
        self.display.DisplayShape(
            self.surf3.face, transparency=0.7, color="BLUE")
        self.display.DisplayShape(self.ray1)
        self.display.DisplayShape(self.ray2)
        self.display.DisplayShape(self.ray3)

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

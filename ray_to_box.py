import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.constants as cnt
from optparse import OptionParser

from OCC.Core.gp import gp_Pnt, gp_Vec, gp_Dir
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Ax3
from OCC.Core.gp import gp_Lin, gp_Elips, gp_Pln, gp_Circ
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepTools import BRepTools_GTrsfModification
from OCC.Core.BRepCheck import BRepCheck_Shell
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Collect, BRepBuilderAPI_LineThroughIdenticPoints
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Common
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCC.Core.BRepLProp import BRepLProp_SLProps
from OCC.Core.Geom import Geom_Line
from OCC.Core.GeomAPI import GeomAPI_IntCS
from OCC.Core.GeomLProp import GeomLProp_SurfaceTool
from OCC.Core.TopoDS import TopoDS_Shape
from OCC.Core.ShapeAlgo import ShapeAlgo_ToolContainer
from OCC.Core.IntAna import IntAna_Line
from OCC.Core.IntCurveSurface import IntCurveSurface_TransitionOnCurve
from OCC.Core.IntPatch import IntPatch_TheSearchInside
from OCC.Extend.ShapeFactory import measure_shape_volume, measure_shape_mass_center_of_gravity
from OCCUtils.Construct import vec_to_dir, dir_to_vec
from OCCUtils.Construct import point_to_vector, vector_to_point
from OCCUtils.Construct import make_plane, make_polygon, make_box

from base import plotocc, Face, set_trf, rot_axs, set_loc


def move_pnt_to_dir(axs=gp_Ax3(), scale=100):
    vec = point_to_vector(axs.Location()) + dir_to_vec(axs.Direction()) * scale
    return vector_to_point(vec)


class TraceSystem (plotocc):

    def __init__(self):
        plotocc.__init__(self, touch=True)
        self.axs = gp_Ax3(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        self.selected_shape = [
            make_box(gp_Pnt(-100, 10, -100), 250, 250, 250),
            make_plane(gp_Pnt(0, 300, 0), gp_Vec(0, 1, 0)),
            self.make_torus(r0=200, r1=50)
        ]
        self.shp = self.make_comp_selcted()
        self.shp.Location(set_loc(ax2=self.axs))

        self.beam1 = gp_Ax3(gp_Pnt(), gp_Dir(0, 1, 0))
        self.beam1.Transform(set_trf(ax2=self.axs))

        self.show_axs_pln(self.beam1)
        self.reflect_beam(self.shp, self.beam1)

        self.show_axs_pln(self.beam1)
        self.display.DisplayShape(self.shp)

    def reflect_beam_multi(self, shpe=TopoDS_Shape(), beam0=gp_Ax3(), tr=0):
        """
        Calculate the reflection/transmission of a beam by shape

        Args:
            shpe (TopoDS_Shape): The reflection shape. Defaults to TopoDS_Shape().
            beam0 (gp_Ax3): Defaults to gp_Ax3().
            tr (int): 
                0 : reflection (Default)
                1 : transmission
                2 : normal on face.

        Returns:
            beam1 [gp_Ax3]: 
        """
        v0 = dir_to_vec(beam0.Direction())
        v1 = dir_to_vec(beam0.XDirection())
        p0 = beam0.Location()
        d0 = beam0.Direction()
        lin = gp_Lin(p0, d0)
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
            print(api.Transition(), p1)
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
            # Reflection
            beam1.Mirror(norm1.Ax2())
            if beam1.Direction().Dot(norm1.Direction()) < 0:
                beam1.ZReverse()
        elif tr == 1:
            # Transmission
            beam1.ZReverse()
            beam1.XReverse()
        elif tr == 2:
            beam1 = gp_Ax3(norm1.Ax2())
        return beam1


if __name__ == "__main__":
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./")
    opt, argc = parser.parse_args(argvs)
    print(argc, opt)

    obj = TraceSystem()
    obj.ShowOCC()

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import scipy.constants as cnt
from optparse import OptionParser

from base import plotocc
from Surface import surf_curv


if __name__ == "__main__":
    argvs = sys.argv
    parser = OptionParser()
    parser.add_option("--dir", dest="dir", default="./")
    opt, argc = parser.parse_args(argvs)
    print(argc, opt)

    obj = plotocc()
    obj.show_axs_pln()
    obj.display.DisplayShape(surf_curv())

    obj.show()

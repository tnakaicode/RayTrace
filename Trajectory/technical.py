import numpy as np
import sys
import os
import platform as pt
import subprocess as subprocess


def end_program():
    sys.exit(0)


def zero_matrix(row, column, datatype):
    pre_matrix = np.zeros((row, column), dtype=datatype)
    matrix = pre_matrix.view(type=np.matrix)
    return matrix

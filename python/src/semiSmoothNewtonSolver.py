from dolfinx import fem, mesh, plot, io, geometry
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
from ufl import ds, dx, grad, inner
import ufl
import matplotlib as mpl
import pyvista
from typing import List, Tuple
from src.visualization import timeDependentVariableToGif, printControlFunction, plot_array
from src.solveStateEquation import solveStateEquation, getSourceTerm
from src.solveAdjointEquation import solveAdjointEquation
from src.tools import getValueOfFunction
from src.ExtremalPoints import ExtremalPoint
from dataclasses import dataclass
import scipy

def computeHesseMatrix(arg):
    pass

def computeProx(arg):
    pass

def computeDifferential(arg):
    pass

def computeSemiNewtonStep(active_set, weights, slope, y_shift, params) -> tuple[np.array, np.array, np.array]:
    N = len(active_set)
    c = 1
    DiffG = np.zeros((N+2*params.d, N+2*params.d))
    x_k = np.zeros((N+2*params.d,))
    x_k[:N] = weights
    x_k[-2*params.d:-params.d] = slope
    x_k[-params.d:] = y_shift
    P_c = np.zeros((N+2*params.d,))
    DP_c = np.zeros((N+2*params.d, N+2*params.d))
    Hf = np.zeros((N+2*params.d, N+2*params.d))
    Df = np.zeros((N+2*params.d,))
    # Compute for all extremal points Ku_i and also Ke_i, K(e_ix), (maybe also Ku)
    for k in range(20):
        # Compute Vector P_c
        # Compute Matrix DP_c(weights, slope, y_shift) (shape (N+2+2, N+2+2))
        # Compute Vector Df(P_c(weights, slope, y_shift))
        # Compute Matrix D^2f(P_c(weights, slope, y_shift))
        DiffG = c * (np.identity((N+2*params.d, N+2*params.d)) - DP_c) + np.matmul(Hf, DP_c)
        d_k = scipy.linalg.solve(DiffG, Df, assume_a="sym")
        x_k += d_k
        continue
    return x_k[:N], x_k[-2*params.d:-params.d], x_k[-params.d:]
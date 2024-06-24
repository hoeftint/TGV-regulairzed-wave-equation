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
from src.tools import getValueOfFunction, buildIterationFunction
from src.ExtremalPoints import ExtremalPoint
from src.HesseMatrix import HesseMatrix, calculateL2InnerProduct
from dataclasses import dataclass
import scipy

def computeProxDifferential(x, active_set, params):
    vector = np.zeros((len(active_set) + 2 * params.d,))
    idx = 0
    while (idx < len(active_set)):
        vector[idx] = 1. if x[idx] > 1 / float(params.newton_c) else 0.
        idx += 1
    while (idx < len(active_set) + 2 * params.d):
        vector[idx] = 1.
        idx += 1
    return np.diag(vector)

def computeDifferential(x, active_set: List[ExtremalPoint], standard_states: List[fem.Function], params):
    s1 = lambda t: buildIterationFunction(t, active_set, x[:len(active_set)], 
                                        x[len(active_set): -params.d], x[-params.d:], params)[0]
    s2 = lambda t: buildIterationFunction(t, active_set, x[:len(active_set)],
                                        x[len(active_set): -params.d], x[-params.d:], params)[1]
    g1 = getSourceTerm(params.x1, params)
    g2 = getSourceTerm(params.x2, params)
    K_u, __ = solveStateEquation([g1, g2], [s1, s2], params)
    phi = [fem.Function(params.V) for _ in K_u]
    for idx in range(len(phi)):
        phi[idx].x.array[:] = 0 * K_u[idx].x.array - params.yd[idx].x.array
    vector = np.zeros((len(active_set) + 2 * params.d,))
    idx = 0
    for extremal in active_set:
        vector[idx] = calculateL2InnerProduct(phi, extremal.state, params)
        idx += 1
    for state in standard_states:
        vector[idx] = calculateL2InnerProduct(phi, state, params)
    return vector

def computeSemiNewtonStep(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
    n = len(active_set) + 2 * params.d
    standard_states = hesse.standard_states
    DiffG = np.zeros((n, n))
    x_k = np.zeros((n,))
    x_k[:len(active_set) - 1] = weights
    x_k[len(active_set) - 1] = 1.
    x_k[len(active_set):-params.d] = slope
    x_k[-params.d:] = y_shift
    DP_c = np.zeros((n, n))
    Df = np.zeros((n,))
    identity = np.identity(n)
    for k in range(10):
        #print(k, 'Newton iterate: ', x_k)
        DP_c = computeProxDifferential(x_k, active_set, params)
        Df = computeDifferential(x_k, active_set, standard_states, params)
        #print('iwdentity: ', identity.shape, 'DP_c: ', DP_c.shape, 'Df: ', Df.shape, 'hesse: ', hesse.matrix.shape)
        DiffG = params.newton_c * (identity - DP_c) + np.matmul(hesse.matrix, DP_c)
        #print(DiffG)
        #print(np.linalg.det(1e5 * DiffG))
        d_k = scipy.linalg.solve(DiffG, Df)
        x_k += d_k
        if np.linalg.norm(d_k) < 1e-5:
            break

    print('Newton solution: ', x_k)
    return x_k[:len(active_set)], x_k[len(active_set): -params.d], x_k[-params.d:]

def solveFinDimProblem(weights, slope, y_shift, params, active_set, yd):
	x0 = np.zeros(len(active_set) + 2 * params.d)
	x0[:len(active_set) - 1] = weights
	x0[len(active_set) - 1] = 0.5
	x0[-2*params.d:-params.d] = slope
	x0[-params.d:] = y_shift
    
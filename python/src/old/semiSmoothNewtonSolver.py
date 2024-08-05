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
from helpers import getValueOfFunction, buildIterationFunction
from src.ExtremalPoints import ExtremalPoint
from src.HesseMatrix import HesseMatrix, calculateL2InnerProduct
from dataclasses import dataclass
import scipy

# Dissertation zu semismooth Newton https://mediatum.ub.tum.de/doc/1241413/1241413.pdf

def computeProxDifferential(x, active_set, params):
    vector = np.ones_like(x)
    idx = 0
    while (idx < len(active_set)):
        vector[idx] = 1. if x[idx] - 1. / float(params.newton_c) >= 0 else 0.
        idx += 1
    return np.diag(vector)

def computeProx(x, active_set, params):
    prox = np.copy(x)
    prox[:len(active_set)] = np.clip((x[:len(active_set)] - 1/float(params.newton_c) 
                                      * np.ones_like(x[:len(active_set)])), a_min=0, a_max=None)
    return prox

def computeDifferential(x, hesse: HesseMatrix, vectorStandardInner, active_set, params):
    vector = np.zeros_like(x)
    vector[:] = hesse.matrix.dot(x) - vectorStandardInner + params.gamma * x
    return vector

def computeObjective(x, active_set, standard_states, params):
    phi = [fem.Function(params.V) for _ in params.yd]
    for idx in range(len(phi)):
        phi[idx].x.array[:] = -params.yd[idx].x.array
        for j, func in enumerate(active_set):
            phi[idx].x.array[:] += x[j] * func.state[idx].x.array
        for j, func in enumerate(standard_states):
            phi[idx].x.array[:] += x[len(active_set) + j] * func[idx].x.array
    sum_points = 0
    for i in range(len(active_set)):
        sum_points += x[i]
    return 0.5 * calculateL2InnerProduct(phi, phi, params) + sum_points + 0.5 * params.gamma * np.linalg.norm(x)**2

def computeSemiNewtonStep(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
    n = len(active_set) + 2 * params.d
    armijoParameter = 0.5
    standard_states = hesse.standard_states
    
    vectorStandardInner = np.zeros((n,))
    for idx, func in enumerate(active_set):
        vectorStandardInner[idx] = func.standardInner
    for idx, func in enumerate(standard_states):
        vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)

    x_k = np.concatenate((weights, slope, y_shift))
    q = np.copy(x_k)
    q[:len(active_set)] = x_k[:len(active_set)] + 1/params.newton_c * np.ones_like(x_k[:len(active_set)])
    P_c = computeProx(q, active_set, params)
    obj = computeObjective(P_c, active_set, standard_states, params)
    print('Objective: ', obj)
    for k in range(params.maxNewtonSteps):
        Df = computeDifferential(q, hesse, vectorStandardInner, active_set, params)
        DP_c = computeProxDifferential(q, active_set, params)
        DG = params.newton_c * (np.identity(n) - DP_c) + np.matmul(hesse.matrix + params.gamma * np.identity(n), DP_c)
        G = params.newton_c * (q - P_c) + Df + params.gamma * q
        dq = scipy.linalg.solve(DG, -G, 'sym')
        sigma = 1
        obj_q = computeObjective(q, active_set, standard_states, params)
        obj_q_damp = computeObjective(q + sigma * dq, active_set, standard_states, params)
        #while obj_q_damp > obj_q and k > 1:
        #    sigma = sigma * armijoParameter
        #    obj_q_damp = computeObjective(q + sigma * dq, active_set, standard_states, params)
        sigma = 1
        q[:] = q + sigma * dq
        P_c = computeProx(q, active_set, params)
        objNew = computeObjective(P_c, active_set, standard_states, params)
        if abs(objNew - obj) < 1e-5 and k > 1:
            break
        obj = objNew
        print(k, ': Objective: ', obj)
        print(k, ': Iteration: ', P_c)

    print('Newton solution: ', P_c)
    return P_c[:len(active_set)], P_c[-2*params.d:-params.d], P_c[-params.d:]

def computeProxGradStep(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
    x_k = np.concatenate((weights, slope, y_shift))
    n = len(active_set) + 2 * params.d
    standard_states = hesse.standard_states
    
    vectorStandardInner = np.zeros((n,))
    for idx, func in enumerate(active_set):
        vectorStandardInner[idx] = func.standardInner
    for idx, func in enumerate(standard_states):
        vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)
    obj = computeObjective(x_k, active_set, standard_states, params)
    q = np.copy(x_k)
    q[:len(active_set)] = x_k[:len(active_set)] + 1/params.newton_c * np.ones_like(x_k[:len(active_set)])
    P_c = computeProx(q, active_set, params)
    theta = 1
    #print(': Objective value:', obj)
    for k in range(50):
        Df = computeDifferential(q, hesse, vectorStandardInner, params)
        G = params.newton_c * (q - P_c) + Df + params.gamma * q
        q = q - theta * G
        theta /= 1.01
        P_c = computeProx(q, active_set, params)
        objNew = computeObjective(P_c, active_set, standard_states, params)
        if abs(objNew - obj) < 1e-8:
            break
        obj = objNew
        print(k, ': Objective value:', obj)
    print(': Objective value:', obj)
    print('Prox solution: ', x_k)
    return P_c[:len(active_set)], P_c[-2*params.d:-params.d], P_c[-params.d:]
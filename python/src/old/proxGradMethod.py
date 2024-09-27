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
from solutionOperators import solveStateEquation, getSourceTerm
from src.tools import getValueOfFunction, buildIterationFunction
from src.ExtremalPoints import ExtremalPoint
from src.HesseMatrix import HesseMatrix, calculateL2InnerProduct
from src.semiSmoothNewtonSolver import computeProxDifferential, computeProx, computeDifferential, computeObjective
from dataclasses import dataclass
import scipy

def phi(x, active_set):
    result = 0
    for idx in range(len(active_set)):
        if x[idx] < 0:
            return np.inf
        result = result + x[idx]
    return result

def checkFilter(x_k_new, F_k_new, theta, filter_set):
    gammaFilter = 7 * 1e-2
    for q in filter_set:
        if (np.max(q - theta) < gammaFilter * np.max(theta)):
            return False
    return True
    
def globalizedSemiSmoothMethod(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
    x_k = np.concatenate((weights, slope, y_shift))
    n = len(active_set) + 2 * params.d
    standard_states = hesse.standard_states
    
    vectorStandardInner = np.zeros((n,))
    for idx, func in enumerate(active_set):
        vectorStandardInner[idx] = func.standardInner
    for idx, func in enumerate(standard_states):
        vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)
    
    Lambda = params.newton_c * np.identity(n)
    invLambda = 1 / params.newton_c * np.identity(n)
    Df_k = computeDifferential(x_k, hesse, vectorStandardInner, active_set, params)
    Df_k_new = np.zeros_like(Df_k)
    u_k = x_k - invLambda.dot(Df_k)
    u_k_new = np.zeros_like(u_k)
    s_k = np.zeros_like(x_k)
    x_k_new = np.zeros_like(x_k)
    prox_k = computeProx(u_k, active_set, params)
    prox_k_new = np.zeros_like(prox_k)
    F_k = x_k - prox_k
    F_k_new = np.zeros_like(F_k)
    DF_k = np.identity(n) - computeProxDifferential(u_k, active_set, params).dot(np.identity(n) - invLambda.dot(hesse.matrix))
    beta = 0.1
    gamma = 0.1
    k = 0
    filter_set = []
    newtonAccepted = True
    theta = np.absolute(F_k)
    while np.linalg.norm(F_k) >= 1e-5:
        if newtonAccepted:
            filter_set.append(theta)
        s_k[:] = scipy.linalg.solve(-DF_k, F_k)
        x_k_new[:] = x_k + s_k
        print('x_k: ', x_k)
        print('x_k_new: ', x_k_new)
        print('s_k: ', s_k)
        if phi(x_k_new, active_set) < np.inf:
            print('Hallo')
            Df_k_new[:] = computeDifferential(x_k_new, hesse, vectorStandardInner, active_set, params)
            u_k_new[:] = x_k_new - invLambda.dot(Df_k_new)
            prox_k_new[:] = computeProx(u_k_new, active_set, params)
            F_k_new[:] = x_k_new - prox_k_new
            theta = np.absolute(F_k_new)
            if (checkFilter(x_k_new, F_k_new, theta, filter_set)):
                x_k[:] = x_k_new
                F_k[:] = F_k_new
                newtonAccepted = True
                filter_set.append(theta)
            else:
                newtonAccepted = False
        else:
            newtonAccepted = False
        if not newtonAccepted:
            sigma = 1
            s_k[:] = -F_k
            obj = computeObjective(x_k, active_set, standard_states, params)
            Delta_k = -Df_k.dot(F_k) + phi(prox_k, active_set) -  phi(x_k, active_set)
            # quasi-Armijo rule as in Algorithm 1: (https://mediatum.ub.tum.de/doc/1289514/669579.pdf)
            while computeObjective(x_k + sigma * s_k, active_set, standard_states, params) > obj + sigma * gamma * Delta_k:
                sigma = sigma * beta
            x_k[:] = x_k + sigma * s_k
            Df_k[:] = computeDifferential(x_k, hesse, vectorStandardInner, params)
            u_k[:] = x_k - invLambda.dot(Df_k)
            prox_k[:] = computeProx(x_k - invLambda.dot(Df_k), active_set, params)
            F_k[:] = x_k - prox_k
            DF_k[:] = np.identity(n) - computeProxDifferential(u_k, active_set, params).dot(np.identity(n) - invLambda.dot(hesse.matrix))
        k = k+1
        print(k, 'Is Newton: ', newtonAccepted, 'Objective: ', computeObjective(x_k, active_set, standard_states, params))
    print('Prox solution:', x_k)
    return x_k[:len(active_set)], x_k[-2*params.d:-params.d], x_k[-params.d:]
            
            
        
    return x_k[:len(active_set)], x_k[-2*params.d:-params.d], x_k[-params.d:]
    
    
def computeProxGradStep(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
    x_k = np.concatenate((weights, slope, y_shift))
    n = len(active_set) + 2 * params.d
    standard_states = hesse.standard_states
    
    vectorStandardInner = np.zeros((n,))
    for idx, func in enumerate(active_set):
        vectorStandardInner[idx] = func.standardInner
    for idx, func in enumerate(standard_states):
        vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)
    
    Lambda = params.newton_c * np.identity(n)
    invLambda = 1 / params.newton_c * np.identity(n)
    Df_k = computeDifferential(x_k, hesse, vectorStandardInner, params)
    prox_k = computeProx(x_k - invLambda.dot(Df_k), active_set, params)
    F_k = x_k - prox_k
    beta = 0.1
    gamma = 0.1
    k = 0
    while np.linalg.norm(F_k) >= 1e-4:
        print(k, ': Objective: ', computeObjective(x_k, active_set, standard_states, params))
        k += 1
        d_k = -F_k
        obj = computeObjective(x_k, active_set, standard_states, params)
        Delta_k = -Df_k.dot(F_k) + phi(prox_k, active_set) -  phi(x_k, active_set)
        sigma = 1
        # quasi-Armijo rule as in Algorithm 1: (https://mediatum.ub.tum.de/doc/1289514/669579.pdf)
        while computeObjective(x_k + sigma * d_k, active_set, standard_states, params) > obj + sigma * gamma * Delta_k:
            sigma = sigma * beta
        x_k = x_k + sigma * d_k
        Df_k = computeDifferential(x_k, hesse, vectorStandardInner, params)
        prox_k = computeProx(x_k - invLambda.dot(Df_k), active_set, params)
        F_k = x_k - prox_k
    print('Prox solution:', x_k)
    return x_k[:len(active_set)], x_k[-2*params.d:-params.d], x_k[-params.d:]
    
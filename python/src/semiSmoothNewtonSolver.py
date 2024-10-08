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
from src.solutionOperators import solveStateEquation, getSourceTerm
from src.helpers import getValueOfFunction, buildIterationFunction, calculateL2InnerProduct
from src.ExtremalPoints import ExtremalPoint
from src.HesseMatrix import HesseMatrix
from dataclasses import dataclass
import scipy

# Dissertation zu semismooth Newton https://mediatum.ub.tum.de/doc/1241413/1241413.pdf

def computeProxDifferential(x, active_set, reg, params):
	vector = np.ones_like(x)
	for idx in range(len(active_set)):
		vector[idx] = 1. if x[idx] - reg[idx] / float(params.newton_c) >= 0 else 0.
	return np.diag(vector)

def computeProx(x, active_set, reg, params):
	prox = np.copy(x)
	prox[:len(active_set)] = np.clip((x[:len(active_set)] - reg / float(params.newton_c)), a_min=0, a_max=None)
	return prox

def computeDifferential(x, hesse: HesseMatrix, vectorStandardInner):
	vector = np.zeros_like(x)
	vector[:] = hesse.matrix.dot(x) - vectorStandardInner
	return vector

def computeObjective(x, active_set, standard_states, params):
	kind = np.array([func.type for func in active_set])
	reg = kind * params.beta - params.alpha * (kind - 1)
	phi = [fem.Function(params.V) for _ in params.yd]
	for idx in range(len(phi)):
		phi[idx].x.array[:] = -params.yd[idx].x.array
		for j, func in enumerate(active_set):
			phi[idx].x.array[:] += x[j] * func.state[idx].x.array
		for j, func in enumerate(standard_states):
			phi[idx].x.array[:] += x[len(active_set) + j] * func[idx].x.array
	sum_points = 0
	for i, func in enumerate(active_set):
		sum_points += x[i] * reg[i]
	return 0.5 * calculateL2InnerProduct(phi, phi, params) + sum_points

def computeSSNStep(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
	n = len(active_set) + 2 * params.d
	standard_states = hesse.standard_states
	
	vectorStandardInner = np.zeros((n,))
	for idx, func in enumerate(active_set):
		vectorStandardInner[idx] = func.standardInner
	for idx, func in enumerate(standard_states):
		vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)

	Id = np.identity(n)
	theta = 1e-9
	kind = np.array([func.type for func in active_set])
	reg = kind * params.beta - params.alpha * (kind - 1)
	u = np.concatenate((weights, slope, y_shift))
	q = u + 1/params.newton_c * np.array([1. if k < len(active_set) else 0. for k in range(n)])
	P_c = computeProx(q, active_set, reg, params)
	#print('Objective: ', obj)
	#print(-1, ': Iteration: ', P_c)
	for k in range(params.maxNewtonSteps):
		Df = computeDifferential(P_c, hesse, vectorStandardInner)
		DP_c = computeProxDifferential(q, active_set, reg, params)
		G = params.newton_c * (q - P_c) + Df
		DG = params.newton_c * (Id - DP_c) + np.matmul(hesse.matrix, DP_c)
		theta /= 2
		dq = scipy.linalg.solve(DG + theta * Id, -G, 'pos')
		q_new = q + dq
		P_c_new = computeProx(q_new, active_set, reg,params)
		
		qDiff = computeObjective(P_c, active_set, standard_states, params) - computeObjective(P_c_new, active_set, standard_states, params)
		while qDiff > 1e-3:
			theta = 2 * theta
			dq = scipy.linalg.solve(DG + theta * Id, -G, 'pos')
			q_new = q + dq
			P_c_new = computeProx(q_new, active_set, reg, params)
			qDiff = computeObjective(P_c, active_set, standard_states, params) - computeObjective(P_c_new, active_set, standard_states, params)
		P_c = P_c_new
		q = q_new
		if abs(qDiff) < 1e-16 and k > 1:
			break
		#print(k, ': Iteration: ', P_c, 'Iteration raw', q)

	#print('Newton solution: ', P_c)
	return P_c[:len(active_set)], P_c[-2 * params.d:-params.d], P_c[-params.d:]

def computeProxGradStep(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
	u_k = np.concatenate((weights, slope, y_shift))
	n = len(active_set) + 2 * params.d
	standard_states = hesse.standard_states
	
	vectorStandardInner = np.zeros((n,))
	for idx, func in enumerate(active_set):
		vectorStandardInner[idx] = func.standardInner
	for idx, func in enumerate(standard_states):
		vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)
	obj = computeObjective(u_k, active_set, standard_states, params)
	q = np.copy(u_k)
	q[:len(active_set)] = u_k[:len(active_set)] + 1/params.newton_c * np.ones_like(u_k[:len(active_set)])
	P_c = computeProx(q, active_set, params)
	theta = 1
	#print(': Objective value:', obj)
	for k in range(params.maxNewtonSteps):
		Df = computeDifferential(q, hesse, vectorStandardInner, params)
		G = params.newton_c * (q - P_c) + Df + params.gamma * q
		q = q - theta * G
		theta /= 1.01
		P_c = computeProx(q, active_set, params)
		objNew = computeObjective(P_c, active_set, standard_states, params)
		if abs(objNew - obj) < 1e-8:
			break
		obj = objNew
		#print(k, ': Objective value:', obj)
	print(': Objective value:', obj)
	print('Prox solution: ', u_k)
	return P_c[:len(active_set)], P_c[-2*params.d:-params.d], P_c[-params.d:]
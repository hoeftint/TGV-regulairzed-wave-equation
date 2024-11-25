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
from src.helpers import getValueOfFunction, buildIterationFunction, computeObjective
from src.ExtremalPoints import ExtremalPoint
from src.HesseMatrix import HesseMatrix, calculateL2InnerProduct
from dataclasses import dataclass
import scipy

def computeDifferential(x, hesse: HesseMatrix, vectorStandardInner, reg):
	vector = np.zeros_like(x)
	vector[:] = hesse.matrix.dot(x) - vectorStandardInner + reg
	return vector

def computeSSNStepWalter(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
	n = len(active_set) + 2 * params.d
	tol = 1e-10
	standard_states = hesse.standard_states
	vectorStandardInner = np.zeros((n,))
	for idx, func in enumerate(active_set):
		vectorStandardInner[idx] = func.standardInner
	for idx, func in enumerate(standard_states):
		vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)
	kind = np.array([point.type for point in active_set], dtype=int)
	Id = np.identity(n)
	theta = 1e-9
	point = np.concatenate((weights, slope, y_shift))
	reg = np.zeros_like(point)
	reg[:len(active_set)] = params.beta * kind - params.alpha * (kind - 1)
	#misfit = computeMisfit(point, active_set, hesse.standard_states, params)
	# WARNING: In Daniels implementation, reg is used for q (not sure why)
	q = point - computeDifferential(point, hesse, vectorStandardInner, reg)
	point = np.copy(q)
	point[:len(active_set)] = np.clip((q[:len(active_set)]), a_min=0, a_max=None)
	for i in range(params.maxNewtonSteps):
		G = q - point + computeDifferential(point, hesse, vectorStandardInner, reg)
		#print('norm G: ',np.linalg.norm(G))#, '\tq: ', q, '\tpoint', point)
		if (np.linalg.norm(G) <= tol):
			break
		vector = np.copy(q)
		vector[-2 *params.d:] = np.ones_like(vector[-2 *params.d:])
		vector = np.array([x >= 0 for x in vector])
		#print('q: ', q, 'vector: ', vector)
		D = np.diag(vector)
		M = Id - D + np.matmul(hesse.matrix, D)
		theta /= 1000
		dq = scipy.linalg.solve(M + theta * Id, G, 'pos')
		qNew = q - dq
		pointNew = np.copy(qNew)
		pointNew[:len(active_set)] = np.clip((qNew[:len(active_set)]), a_min=0, a_max=None)
		qDiff = computeObjective(pointNew, active_set, standard_states, hesse, params, vectorStandardInner) - computeObjective(point, active_set, standard_states, hesse, params, vectorStandardInner)
		while qDiff > 1e-3:
			theta = 2*theta
			dq = scipy.linalg.solve(M + theta * Id, G, 'pos')
			qNew = q - dq
			pointNew = np.copy(qNew)
			pointNew[:len(active_set)] = np.clip((qNew[:len(active_set)]), a_min=0, a_max=None)
			qDiff = computeObjective(pointNew, active_set, standard_states, hesse, params, vectorStandardInner) - computeObjective(point, active_set, standard_states, hesse, params, vectorStandardInner)
		q = qNew
		point = pointNew

	return (point[:len(active_set)], point[-2 * params.d:-params.d], point[-params.d:])

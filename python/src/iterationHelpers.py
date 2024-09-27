from dolfinx import fem
import numpy as np
import scipy
from ufl import ds, dx, grad, inner, dot
from mpi4py import MPI
from typing import List
from src.solutionOperators import solveStateEquation, getSourceTerm, buildControlFunction, solveAdjointEquation, buildControlFunctionAdjoint
from src.ExtremalPoints import ExtremalPoint
from src.HesseMatrix import HesseMatrix
from src.helpers import getValueOfFunction, linCombFunctionLists, buildIterationFunction
from src.visualization import timeDependentVariableToGif, plot_array, plot_function
from typing import List

def calculateDiscreteGradient(active_set: List[ExtremalPoint], weights, slope, y_shift, hesse, params):
	u1 = lambda t: buildIterationFunction(t, active_set, weights, slope, y_shift, params)[0]
	u2 = lambda t: buildIterationFunction(t, active_set, weights, slope, y_shift, params)[1]
	g1 = getSourceTerm(params.x1, params)
	g2 = getSourceTerm(params.x2, params)
	control = buildControlFunction([g1, g2], [u1, u2], params)
	if not params.useDummy:
		K_u = solveStateEquation(control, params)
	else:
		K_u = control
	phi = [fem.Function(params.V) for _ in K_u]
	for idx in range(len(phi)):
		phi[idx].x.array[:] = K_u[idx].x.array - params.yd[idx].x.array
	if not params.useDummy:
		adjointState = solveAdjointEquation(phi, params)
	else:
		adjointState = phi
	result = buildControlFunctionAdjoint([g1, g2], adjointState, params)
	return result

def calculateFirstDual(active_set: List[ExtremalPoint], weights, slope, y_shift, standardFirstDuals, params):
	g1 = getSourceTerm(params.x1, params)
	g2 = getSourceTerm(params.x2, params)
	'''
	if not isinstance(params.yd_firstDual, np.ndarray):
		dual = np.ndarray((len(params.yd_adjoint), params.d), dtype=np.float64)
		for idx, func in enumerate(params.yd_adjoint):
			energy_form = fem.form(inner(g1, func) * dx)
			energy_local = fem.assemble_scalar(energy_form)
			energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
			dual[idx,0] = energy_global
			energy_form = fem.form(inner(g2, func) * dx)
			energy_local = fem.assemble_scalar(energy_form)
			energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
			dual[idx,1] = energy_global
		params.yd_firstDual = integrateVectorFunction(dual, params)'''
	primitiveState = []#linCombFunctionLists(0, [], 1, params.yd_firstDual, params)
	for idx, func in enumerate(active_set):
		primitiveState = linCombFunctionLists(1, primitiveState, weights[idx], func.firstDual, params)
	for idx in range(params.d):
		primitiveState = linCombFunctionLists(1, primitiveState, slope[idx], standardFirstDuals[idx], params)
	for idx in range(params.d):
		primitiveState = linCombFunctionLists(1, primitiveState, y_shift[idx], standardFirstDuals[params.d + idx], params)
	adjointPrimitiveState = solveAdjointEquation(primitiveState, params)
	result = buildControlFunctionAdjoint([g1, g2], adjointPrimitiveState, params)
	'''
	firstDual = linCombFunctionLists(1, adjointPrimitiveState, -1, params.yd_firstDual, params)
	for idx, func in enumerate(firstDual):
		energy_form = fem.form(inner(g1, func) * dx)
		energy_local = fem.assemble_scalar(energy_form)
		energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
		result[idx,0] = energy_global
		energy_form = fem.form(inner(g2, func) * dx)
		energy_local = fem.assemble_scalar(energy_form)
		energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
		result[idx,1] = energy_global
	'''
	return result

def integrateVectorFunction(function_array, params):
	#integrated_function = scipy.integrate.cumulative_trapezoid(function_array, dx=params.dt, initial=0)
	integrated_function = np.zeros_like(function_array)
	timePoints = np.linspace(0, params.T, num=len(integrated_function[:,0]))
	integrated_function[:, 0] = scipy.integrate.cumulative_simpson(function_array[:,0], x=timePoints, initial=0)	
	integrated_function[:, 1] = scipy.integrate.cumulative_simpson(function_array[:,1], x=timePoints, initial=0)
	return integrated_function
	
def pruneActiveSet(active_set: List[ExtremalPoint], weights, threshold):
	newPoint = active_set[-1]
	for idx in range(len(active_set) - 1):
		if active_set[idx].type == newPoint.type and active_set[idx].x_0 == newPoint.x_0:
			if weights[idx] <= threshold:
				continue
			new_sigma = weights[-1] * newPoint.sigma + weights[idx] * active_set[idx].sigma
			active_set[idx].sigma = new_sigma / np.linalg.norm(new_sigma)
			weights[-1] = 0
			break
	new_active_set = []
	new_weights = []
	for idx, func in enumerate(active_set):
		if weights[idx] > threshold:
			new_active_set.append(func)
			new_weights.append(weights[idx])
	return new_active_set, np.array(new_weights)
	
def getIdxMax(value_array, derivative_array, active_set, type):
	exept_idcs = [func.idx for func in active_set if func.type == type]#func.type == type]
	norm_array = np.linalg.norm(value_array, axis=1)
	mask = np.zeros(norm_array.size, dtype=bool)
	mask[exept_idcs] = True
	mask[-1] = True
	mask[0] = True
	clean_array = np.ma.array(norm_array, mask=mask)
	if clean_array.size == 0:
		return -1
	idx = np.argmax(clean_array)
	return idx

def showNonStationarity(discreteDf, active_set, params):
	timePoints = np.linspace(0, params.T, num=len(discreteDf[:,0]))
	firstConditionValues = np.zeros(2 * params.d)
	firstConditionValues[:2] = integrateVectorFunction(discreteDf * timePoints[:, np.newaxis], params)[-1, :]
	firstConditionValues[2:] = integrateVectorFunction(discreteDf, params)[-1, :]
	print('First conditions: ', firstConditionValues, ' (should be close to 0)')
	secondConditionValues = np.zeros(len(active_set))
	for idx, func in enumerate(active_set):
		array = np.ndarray((len(timePoints), params.d))
		array[:,0] = np.array([func.value(t)[0] for t in timePoints])
		array[:,1] = np.array([func.value(t)[1] for t in timePoints])
		secondConditionValues[idx] = scipy.integrate.simpson(np.sum(discreteDf * array, axis=1), x=timePoints) + 1
	print('Second conditions: ', secondConditionValues, ' (should be greater or equal than 0)')
	


def calculateDiscreteGradient2(active_set: List[ExtremalPoint], weights, slope, y_shift, hesse: HesseMatrix, params):
	residualState = linCombFunctionLists(-1, params.yd, 0, [], params)
	for idx, func in enumerate(active_set):
		residualState = linCombFunctionLists(1, residualState, weights[idx], func.state, params)
	for idx in range(params.d):
		residualState = linCombFunctionLists(1, residualState, slope[idx], hesse.standard_states[idx], params)
	for idx in range(params.d):
		residualState = linCombFunctionLists(1, residualState, y_shift[idx], hesse.standard_states[params.d + idx], params)
	g1 = getSourceTerm(params.x1, params)
	g2 = getSourceTerm(params.x2, params)
	adjointState = solveAdjointEquation(residualState, params)
	result = buildControlFunctionAdjoint([g1, g2], adjointState, params)
	return result

def calculateDiscreteGradient3(active_set: List[ExtremalPoint], weights, slope, y_shift, hesse, params):
	adjointState = linCombFunctionLists(-1, params.yd_adjoint, 0, [], params)
	for idx, func in enumerate(active_set):
		adjointState = linCombFunctionLists(1, adjointState, weights[idx], func.adjoint, params)
	for idx in range(params.d):
		adjointState = linCombFunctionLists(1, adjointState, slope[idx], hesse.standard_adjoints[idx], params)
	for idx in range(params.d):
		adjointState = linCombFunctionLists(1, adjointState, y_shift[idx], hesse.standard_adjoints[params.d + idx], params)
	g1 = getSourceTerm(params.x1, params)
	g2 = getSourceTerm(params.x2, params)
	result = buildControlFunctionAdjoint([g1, g2], adjointState, params)
	return result

def calculateDiscreteGradient4(active_set: List[ExtremalPoint], weights, slope, y_shift, hesse, params):
	g1 = getSourceTerm(params.x1, params)
	g2 = getSourceTerm(params.x2, params)
	result = -buildControlFunctionAdjoint([g1, g2], params.yd_adjoint, params)
	for idx, func in enumerate(active_set):
		result += weights[idx] * buildControlFunctionAdjoint([g1, g2], func.adjoint, params)
	for idx in range(params.d):
		result += slope[idx] * buildControlFunctionAdjoint([g1, g2], hesse.standard_adjoints[idx], params)
	for idx in range(params.d):
		result += y_shift[idx] * buildControlFunctionAdjoint([g1, g2], hesse.standard_adjoints[params.d + idx], params)
	return result
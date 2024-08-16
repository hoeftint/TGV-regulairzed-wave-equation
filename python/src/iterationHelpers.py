from src.solveStateEquation import solveStateEquation, getSourceTerm, buildControlFunction
from src.solveAdjointEquation import solveAdjointEquation
from src.ExtremalPoints import ExtremalPoint
from dolfinx import fem
from src.helpers import getValueOfFunction, linCombFunctionLists
import numpy as np
import scipy
from ufl import ds, dx, grad, inner, dot
from src.visualization import timeDependentVariableToGif, plot_array, plot_function
from mpi4py import MPI
from typing import List

def calculateDiscreteGradient(active_set: List[ExtremalPoint], weights, slope, y_shift, standard_adjoints, params):
	#timeDependentVariableToGif(K_u, filename="output/iteration_state.gif", varname="state", slowMoFactor=2, T=params.T)
	#phi = [fem.Function(params.V) for _ in K_u]
	#for idx in range(len(phi)):
	#	phi[idx].x.array[:] = K_u[idx].x.array - params.yd[idx].x.array
	#adjointState = solveAdjointEquation(phi, params)
	adjointState = linCombFunctionLists(-1, params.yd_adjoint, 0, [], params)
	for idx, func in enumerate(active_set):
		adjointState = linCombFunctionLists(1, adjointState, weights[idx], func.adjoint, params)
	for idx in range(params.d):
		adjointState = linCombFunctionLists(1, adjointState, slope[idx], standard_adjoints[idx], params)
	for idx in range(params.d):
		adjointState = linCombFunctionLists(1, adjointState, y_shift[idx], standard_adjoints[params.d + idx], params)
	#timeDependentVariableToGif(K_u, 'output/current_state.gif')
	#timeDependentVariableToGif(phi, 'output/difference.gif')
	#timeDependentVariableToGif(phi, 'output/difference_adjoint.gif')
	#timeDependentVariableToGif(phi, filename="output/phi_state.gif", varname="state", slowMoFactor=2, T=params.T)
	result = np.ndarray((len(adjointState), params.d), dtype=np.float64)
	#timeDependentVariableToGif(adjointState, filename="output/adjoint_state.gif", varname="state", slowMoFactor=2, T=params.T)
	g1 = getSourceTerm(params.x1, params)
	g2 = getSourceTerm(params.x2, params)
	for idx, func in enumerate(adjointState):
		energy_form = fem.form(inner(g1, func) * dx)
		energy_local = fem.assemble_scalar(energy_form)
		energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
		result[idx,0] = energy_global
		energy_form = fem.form(inner(g2, func) * dx)
		energy_local = fem.assemble_scalar(energy_form)
		energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
		result[idx,1] = energy_global
		#adjointValues[idx] = getValueOfFunction(params.V, func, [params.x1, params.x2])[:, 0]
	#print('Adjoint values',adjointValues)
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
	firstDual = linCombFunctionLists(1, adjointPrimitiveState, -1, params.yd_firstDual, params)
	result = np.ndarray((len(adjointPrimitiveState), params.d), dtype=np.float64)
	for idx, func in enumerate(firstDual):
		energy_form = fem.form(inner(g1, func) * dx)
		energy_local = fem.assemble_scalar(energy_form)
		energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
		result[idx,0] = energy_global
		energy_form = fem.form(inner(g2, func) * dx)
		energy_local = fem.assemble_scalar(energy_form)
		energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
		result[idx,1] = energy_global
	return result

def integrateVectorFunction(function_array, params):
	#integrated_function = scipy.integrate.cumulative_trapezoid(function_array, dx=params.dt, initial=0)
	integrated_function = np.zeros_like(function_array)
	timePoints = np.linspace(0, params.T, num=len(integrated_function[:,0]))
	integrated_function[:, 0] = scipy.integrate.cumulative_simpson(function_array[:,0], x=timePoints, initial=0)	
	integrated_function[:, 1] = scipy.integrate.cumulative_simpson(function_array[:,1], x=timePoints, initial=0)
	integrated_norm = np.zeros_like(function_array)
	#integrated_norm[:, 0] = scipy.integrate.cumulative_simpson(np.absolute(function_array[:,0]), x=timePoints, initial=0)
	#integrated_norm[:, 1] = scipy.integrate.cumulative_simpson(np.absolute(function_array[:,1]), x=timePoints, initial=0)
	#num_error = integrated_function[-1]
	#integrated_function[:, 0] = integrated_function[:, 0] - integrated_norm[:, 0] / integrated_norm[-1, 0] * num_error[0]
	#integrated_function[:, 1] = integrated_function[:, 1] - integrated_norm[:, 1] / integrated_norm[-1, 1] * num_error[1]
	return integrated_function
	
def pruneActiveSet(active_set, weights, threshold):
	new_active_set = []
	new_weights = []
	for idx, func in enumerate(active_set):
		if weights[idx] > threshold:
			new_active_set.append(func)
			new_weights.append(weights[idx])
	return new_active_set, np.array(new_weights)
	
def getIdxMax(value_array, active_set, type):
	exept_idcs = [func.idx for func in active_set if func.type == type]
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
    
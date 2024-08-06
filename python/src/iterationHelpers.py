from src.solveStateEquation import solveStateEquation, getSourceTerm, buildControlFunction
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
	adjointState = linCombFunctionLists(-1, params.adjoint_yd, 0, [], params)
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

def integrateVectorFunction(function_array, params):
	#integrated_function = scipy.integrate.cumulative_trapezoid(function_array, dx=params.dt, initial=0)
	integrated_function = scipy.integrate.cumulative_trapezoid(function_array, dx=params.dt, initial=0)
	return integrated_function
	"""
	integrated_function = np.zeros((len(function), params.d), dtype=np.float64)
	integrated_function[0] = np.zeros((params.d,))
	for idx in range(len(function) - 1):
		integrated_function[idx + 1] = integrated_function[idx] + function[idx]
	integrated_function *= params.dt * params.T
	"""
	
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
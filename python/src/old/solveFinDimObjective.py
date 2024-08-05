import numpy as np
from mystic import solvers
from src.solveStateEquation import solveStateEquation, getSourceTerm
from dolfinx import fem
from helpers import buildIterationFunction

def finDimObjective(weights, slope, y_shift, params, active_set, yd):
	u = lambda t: buildIterationFunction(t, active_set, weights, slope, y_shift, params)

	s1 = lambda t: u(t)[0]
	s2 = lambda t: u(t)[1]
	
	g1 = getSourceTerm(params.x1, params)
	g2 = getSourceTerm(params.x2, params)
	K_u, __ = solveStateEquation([g1, g2], [s1, s2], params)
	params.solverIteration += 1
	#print(params.solverIteration)
	phi = [fem.Function(params.V) for _ in K_u]
	for idx in range(len(phi)):
		phi[idx].x.array[:] = K_u[idx].x.array - yd[idx].x.array
	squared = np.zeros_like(phi[0].x.array)
	for func in phi:
		squared += np.sqrt(func.x.array)
	squared /= params.T/params.dt
	return (0.5 * np.sqrt(np.mean(squared)) + np.sum(weights))

def solveFinDimProblem(weights, slope, y_shift, params, active_set, yd):
	x0 = np.zeros(len(active_set) + 2 * params.d)
	x0[:len(active_set) - 1] = weights
	x0[len(active_set) - 1] = 0.5
	x0[-2*params.d:-params.d] = slope
	x0[-params.d:] = y_shift

	bounds = []
	for _ in range(len(active_set)):
		bounds.append((0, np.inf))
	for _ in range(2 * params.d):
		bounds.append((-np.inf, np.inf))
	cost = lambda x: finDimObjective(x[:len(active_set)], x[-2*params.d:-params.d], x[-params.d:], params, active_set, yd)
	solution = solvers.fmin(cost, x0=x0, bounds=bounds, maxiter=10, maxfun=2)
	params.solverIterations = 0
	new_weights = solution[:len(active_set)]
	new_slope = solution[-2*params.d:-params.d]
	new_y_shift = solution[-2:]
	return new_weights, new_slope, new_y_shift
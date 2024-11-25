from dolfinx import fem, geometry
import dolfinx
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
import numpy as np
from ufl import dx, grad, inner
import ufl
from typing import List

def getSourceTerm(point, params) -> fem.Function:
	g = fem.Function(params.V)
	g.interpolate(lambda x: np.clip(1 / (np.abs(params.mollify_const) * np.sqrt(np.pi)) *
					np.exp(-(((x[0] - point[0])**2 + (x[1] - point[1])**2) / (params.mollify_const**2))), a_max=None, a_min=0))
	return g

def getSourceTerm2(point_list, params) -> fem.Function:
	g = fem.Function(params.V)
	g.x.array[:] = 2 * np.zeros_like(g.x.array)
	print(g.x.array.shape)
	mesh = params.msh
	mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
	boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
	print(point_list)
	points = np.zeros((len(point_list), 3))
	for idx, point in enumerate(point_list):
		temp = np.array([point[0], point[1], 0])
		points[idx] = temp

	bb_tree = geometry.bb_tree(mesh, dim=mesh.topology.dim)

	cells = []
	points_on_proc = []

	cell_candidates = geometry.compute_collisions_points(bb_tree, points)
	
	# Choose one of the cells that contains the point
	colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)
	print(colliding_cells)

	for idx, point in enumerate(points):
		if len(colliding_cells.links(idx)) > 0:
			points_on_proc.append(point)
			cells.append(colliding_cells.links(idx)[0])

	print(cells)
	print(points_on_proc)
	points_on_proc = np.array(points_on_proc, dtype=np.float64)
	#u_values = g.eval(points_on_proc, cells)
	return g

def buildControlFunction(sources, signals, params):
	control = []
	interval = np.linspace(0, params.T, int(params.T / params.dt))
	for t in interval:
		control_step = fem.Function(params.V)
		control_step.x.array[:] = signals[0](t) * sources[0].x.array + signals[1](t) * sources[1].x.array
		control.append(control_step)
	return control

def buildControlFunctionAdjoint(sources, control, params):
	signals = np.ndarray((len(control), params.d))
	for idx, func in enumerate(control):
		for j in range(params.d):
			energy_form = fem.form(inner(sources[j], func) * dx)
			energy_local = fem.assemble_scalar(energy_form)
			energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
			signals[idx,j] = energy_global
	return signals

'''
def solveStateEquation(control: List[fem.Function], params) -> List[fem.Function]:
	uStart = fem.Function(params.V)
	uStart.interpolate(lambda x : np.zeros(x[0].shape))
	solution = [uStart]
	u0 = fem.Function(params.V)
	u0.interpolate(lambda x : np.zeros(x[0].shape))
	u1 = fem.Function(params.V)
	u1.interpolate(lambda x : np.zeros(x[0].shape))
	u2 = fem.Function(params.V)
	u2.interpolate(lambda x : np.zeros(x[0].shape))

	u = ufl.TrialFunction(params.V)
	v = ufl.TestFunction(params.V)
	g = fem.Function(params.V)
	g.x.array[:] = control[0].x.array

	# diffusion
	#a = 3 * inner(u,  v) * dx + 2 * (params.dt * params.waveSpeed**2) * inner(grad(u),grad(v)) * dx
	#L = 4 * inner(u2, v)*dx - inner(u1, v) * dx + 2 * params.dt * inner(g, v) * dx

	# wave
	#a = inner(u,  v) * dx + (params.dt**2 * params.waveSpeed**2) * inner(grad(u),grad(v)) * dx
	a = 2 * inner(u,  v) * dx + (params.dt**2 * params.waveSpeed**2) * inner(grad(u),grad(v)) * dx
	# backward finite difference
	#L = 2 * inner(u2, v)*dx - inner(u1, v)*dx + (params.dt**2) * inner(g, v) * dx
	L = 5 * inner(u2, v)*dx - 4 * inner(u1, v)*dx + inner(u0, v)*dx + params.dt**2 * inner(g, v) * dx

	interval = np.linspace(0, params.T, int(params.T / params.dt))
	for idx in range(len(interval)):
		g.x.array[:] = control[idx].x.array
		problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		u = problem.solve()
		solution.append(u)
		u0.x.array[:] = u1.x.array
		u1.x.array[:] = u2.x.array
		u2.x.array[:] = u.x.array
	return solution

def solveAdjointEquation(control: List[fem.Function], params, bcs=[]):
	pStart = fem.Function(params.V)
	pStart.interpolate(lambda x : np.zeros(x[0].shape))
	solution = [pStart]
	pT = fem.Function(params.V)
	pT.interpolate(lambda x : np.zeros(x[0].shape))
	pT1 = fem.Function(params.V)
	pT1.interpolate(lambda x : np.zeros(x[0].shape))
	pT2 = fem.Function(params.V)
	pT2.interpolate(lambda x : np.zeros(x[0].shape))
	p = ufl.TrialFunction(params.V)
	v = ufl.TestFunction(params.V)
	g = fem.Function(params.V)
	g.x.array[:] = control[0].x.array
 
	# diffusion
	#a = 3 * inner(p,  v) * dx + 2 * (params.dt * params.waveSpeed**2) * inner(grad(p),grad(v)) * dx
	#L = 4 * inner(pT, v)*dx - inner(pT1, v)*dx + 2 * params.dt * inner(g, v) * dx

	# wave
	a =  2 * inner(p,  v) * dx + (params.dt**2 * params.waveSpeed**2) * inner(grad(p),grad(v)) * dx
	L = 5 * inner(pT, v)*dx - 4 * inner(pT1, v) * dx + inner(pT2, v)*dx + params.dt**2 * inner(g, v) * dx
	biggestIdx = int(params.T/params.dt)
	for idx in range(biggestIdx):
		g.x.array[:] = control[biggestIdx - idx].x.array
		problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		p = problem.solve()
		solution.append(p)
		pT2.x.array[:] = pT1.x.array
		pT1.x.array[:] = pT.x.array
		pT.x.array[:] = p.x.array
	solution.reverse()
	return solution
'''
def solveStateEquation(control: List[fem.Function], params) -> List[fem.Function]:
	solution = []
	# Set up time integration scheme:
	u = fem.Function(params.V)
	u_old = fem.Function(params.V)
	u_mid = 0.5 * (u + u_old)
	udot_old = fem.Function(params.V)
	u_old.interpolate(lambda x: np.zeros_like(x[0]))
	udot_old.interpolate(lambda x: np.zeros_like(x[0]))
	v = ufl.TestFunction(params.V)
	udot = 2/params.dt *u - 2/params.dt *u_old - udot_old
	uddot = (udot - udot_old)/params.dt

	# Weak problem residual for wave equation:
	g = fem.Function(params.V)
	g.x.array[:] = control[0].x.array
	g_old = fem.Function(params.V)
	g_old.interpolate(lambda x: np.zeros_like(x[0]))
	g_mid = 0.5 * (g + g_old)
	res = inner(uddot,v) * dx + (params.waveSpeed**2) * inner(grad(u_mid),grad(v)) * dx - inner(g_mid , v) * dx


	# Time stepping loop:
	rhsForm = -res
	lhsForm = ufl.derivative(res, u)
	for i in range(len(control)):
		temp = fem.Function(params.V)
		temp.x.array[:] = u_old.x.array
		solution.append(temp)
		problem = LinearProblem(lhsForm, rhsForm, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
		du = problem.solve()
		u.x.array[:] = u.x.array + du.x.array
		udot_old.x.array[:] =  2/params.dt *u.x.array - 2/params.dt *u_old.x.array - udot_old.x.array
		u_old.x.array[:] = u.x.array
		if i < len(control) - 1:
			g_old.x.array[:] = g.x.array
			g.x.array[:] = control[i + 1].x.array
	temp = fem.Function(params.V)
	temp.x.array[:] = u_old.x.array
	solution.append(temp)
	return solution

def solveAdjointEquation(control: List[fem.Function], params) -> List[fem.Function]:
	control_rev = []
	control_rev[:] = list(reversed(control))
	return list(reversed(solveStateEquation(control_rev, params)))[:-1]

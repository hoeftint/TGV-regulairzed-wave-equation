from dolfinx import fem, mesh, plot, io
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
from typing import List
from ufl import ds, dx, grad, inner
import ufl
from src.solveStateEquation import solveStateEquation

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
    c = (params.dt**2 * params.waveSpeed**2)
    g = fem.Function(params.V)
    g.x.array[:] = control[0].x.array
    a =  2 * inner(p,  v) * dx + c * inner(grad(p),grad(v)) * dx
    L = 5 * inner(pT, v)*dx -4 * inner(pT1, v) * dx + inner(pT2, v)*dx + params.dt**2 * inner(g, v) * dx
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
    
'''def solveAdjointEquation(control: List[fem.Function], params, bcs=[]) -> List[fem.Function]:
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
    c = (params.dt * params.waveSpeed**2)
    g = fem.Function(params.V)
    g.x.array[:] = control[0].x.array
    a =  inner(p,  v) * dx + c * inner(grad(p),grad(v)) * dx
    L = inner(pT1, v) * dx  + params.dt * inner(g, v) * dx
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
    return solution'''
        
'''control.reverse()
    pT = fem.Function(params.V)
    pT.interpolate(lambda x : np.zeros(x[0].shape))
    pT1 = fem.Function(params.V)
    pT1.interpolate(lambda x : np.zeros(x[0].shape))
    solution = [pT]
    p = ufl.TrialFunction(params.V)
    v = ufl.TestFunction(params.V)
    c = (params.dt**2 * params.waveSpeed**2)
    g = fem.Function(params.V)
    g.x.array[:] = control[0].x.array
    a =  inner(p,  v) * dx + c * inner(grad(p),grad(v)) * dx
    L = 2 * inner(pT, v) * dx - inner(pT1, v) * dx + c * inner(g, v) * dx

    interval = np.linspace(0, params.T, int(params.T / params.dt))
    for idx in range(len(interval)):
        g.x.array[:] = control[idx].x.array
        problem = LinearProblem(a, L, bcs=bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        p = problem.solve()
        solution.append(p)
        pT1.x.array[:] = pT.x.array
        pT.x.array[:] = p.x.array
        
    solution.reverse()
    control.reverse()'''
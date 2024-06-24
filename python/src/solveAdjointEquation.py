from dolfinx import fem, mesh, plot, io
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
from typing import List
from ufl import ds, dx, grad, inner
import ufl

def solveAdjointEquation(V: fem.FunctionSpace, control: List[fem.Function], params, bcs=[]):
    control.reverse()
    pT = fem.Function(V)
    pT.interpolate(lambda x : np.zeros(x[0].shape))
    pT1 = fem.Function(V)
    pT1.interpolate(lambda x : np.zeros(x[0].shape))
    solution = [pT]

    p = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    c = (params.dt**2 * params.waveSpeed**2)
    g = fem.Function(V)
    g.x.array[:] = control[0].x.array
    a = inner(p,  v) * dx + c * inner(grad(p),grad(v)) * dx
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
    control.reverse()
    return solution
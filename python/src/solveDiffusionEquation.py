from dolfinx import fem
import numpy as np
from ufl import dx, grad, inner
from dolfinx.fem.petsc import LinearProblem
import ufl
from typing import List

def solveDiffEquation(control: List[fem.Function], params) -> List[fem.Function]:
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
    c = (params.dt * params.waveSpeed**2)
    g = fem.Function(params.V)
    g.x.array[:] = control[0].x.array
    a = 3 * inner(u,  v) * dx + 2 * c * inner(grad(u),grad(v)) * dx
    # backward finite difference
    L = 4 * inner(u1, v)*dx - inner(u0, v)*dx + 2 * params.dt * inner(g, v) * dx

    interval = np.linspace(0, params.T, int(params.T / params.dt))
    for idx in range(len(interval)):
        g.x.array[:] = control[idx].x.array
        problem = fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
        solution.append(u)
        u0.x.array[:] = u1.x.array
        u1.x.array[:] = u2.x.array
        u2.x.array[:] = u.x.array
    
    return solution


def solveAdjointDiffEquation(control: List[fem.Function], params, bcs=[]) -> List[fem.Function]:
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
    a =  3 * inner(p,  v) * dx + 2 * c * inner(grad(p),grad(v)) * dx
    L = 4 * inner(pT1, v) * dx - inner(pT2, v)*dx + 2 * params.dt * inner(g, v) * dx
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
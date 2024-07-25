from dolfinx import fem, mesh, plot, io
from dolfinx.fem.petsc import LinearProblem
from petsc4py.PETSc import ScalarType
from mpi4py import MPI
import numpy as np
from ufl import ds, dx, grad, inner
import ufl
import matplotlib as mpl
import pyvista

def getSourceTerm(point, params) -> fem.Function:
    g = fem.Function(params.V)
    #g.interpolate(lambda x: np.minimum(1 / (np.abs(alpha) * np.sqrt(np.pi)) *
    #                      np.exp(-(((x[0] - point[0])**2 + (x[1] - point[1])**2) / (alpha**2))), 1))
    g.interpolate(lambda x: 1 / (np.abs(params.mollify_const) * np.sqrt(np.pi)) *
                    np.exp(-(((x[0] - point[0])**2 + (x[1] - point[1])**2) / (params.mollify_const**2))))
    return g

def buildControlFunction(sources, signals, params):
    control = []
    interval = np.linspace(0, params.T, int(params.T / params.dt))
    for t in interval:
        control_step = fem.Function(params.V)
        control_step.x.array[:] = signals[0](t) * sources[0].x.array + signals[1](t) * sources[1].x.array
        control.append(control_step)
    return control

def solveStateEquation(control, params):
    u0 = fem.Function(params.V)
    u0.interpolate(lambda x : np.zeros(x[0].shape))
    u1 = fem.Function(params.V)
    u1.interpolate(lambda x : np.zeros(x[0].shape))
    solution = [u1]

    u = ufl.TrialFunction(params.V)
    v = ufl.TestFunction(params.V)
    c = (params.dt**2 * params.waveSpeed**2)
    g = fem.Function(params.V)
    g.x.array[:] = control[0].x.array
    a = inner(u,  v) * dx + c * inner(grad(u),grad(v)) * dx
    # backward finite difference
    L = 2*inner(u1, v)*dx - inner(u0, v)*dx + c * inner(g, v) * dx

    interval = np.linspace(0, params.T, int(params.T / params.dt))
    for idx in range(len(interval)):
        g.x.array[:] = control[idx].x.array
        problem = LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u = problem.solve()
        solution.append(u)
        u0.x.array[:] = u1.x.array
        u1.x.array[:] = u.x.array
    
    return solution
from dolfinx import fem, mesh, geometry
import numpy as np
from typing import List
from mpi4py import MPI
from ufl import dx, inner

def linCombFunctionLists(x: np.float64, list1: List[fem.Function], 
                            y: np.float64, list2: List[fem.Function], params) -> List[fem.Function]:
    if (len(list1) != len(list2) and len(list1) > 0 and len(list2) > 0):
        raise ValueError(f"Both lists must have the same length, currently len(list1)={len(list1)}, len(list2)={len(list2)}")
    if (len(list1) == 0 and len(list2) == 0):
        raise ValueError("Both lists are empty")
    comb = [fem.Function(params.V) for _ in list1]
    if len(list1) == 0:
        for idx in range(len(list2)):
            comb[idx].x.array[:] = y * list2[idx].x.array
    elif len(list2) == 0:
        for idx in range(len(list1)):
            comb[idx].x.array[:] = x * list1[idx].x.array
    else:
        for idx in range(len(list1)):
            comb[idx].x.array[:] = x * list1[idx].x.array + y * list2[idx].x.array
    return comb

def getValueOfFunction(V: fem.FunctionSpace, function: fem.Function, point_list: List[tuple[float, float]]):
    msh = V.mesh
    msh.topology.create_connectivity(msh.topology.dim-1, msh.topology.dim)
    boundary_facets = mesh.exterior_facet_indices(msh.topology)
    points = np.zeros((len(point_list), 3))
    for idx, point in enumerate(point_list):
        temp = np.array([point[0], point[1], 0])
        points[idx] = temp

    u_values = []
    bb_tree = geometry.bb_tree(msh, dim=msh.topology.dim)

    cells = []
    points_on_proc = []

    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(msh, cell_candidates, points)
    
    for idx, point in enumerate(points):
        if len(colliding_cells.links(idx)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(idx)[0])
          
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    u_values = function.eval(points_on_proc, cells)
    return (u_values[:])

def buildIterationFunction(t, active_set, weights: np.ndarray, slope: np.ndarray, y_shift: np.ndarray, params):
    value = np.zeros_like(slope)
    for idx, func in enumerate(active_set):
        value += weights[idx] * func.value(t)
    return_value = value + np.array(slope) * t + np.array(y_shift)
    return return_value

def calculateL2InnerProduct(firstState: List[fem.Function], secondState: List[fem.Function], params) -> np.float64:
    sum = 0
    for idx, __ in enumerate(firstState):
        energy_form = fem.form(inner(firstState[idx], secondState[idx]) * dx)
        energy_local = fem.assemble_scalar(energy_form)
        energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
        sum += energy_global
    return sum * params.dt * params.T

def computeObjective(x, active_set, standard_states, params):
    phi = [fem.Function(params.V) for _ in params.yd]
    for idx in range(len(phi)):
        phi[idx].x.array[:] = -params.yd[idx].x.array
        for j, func in enumerate(active_set):
            phi[idx].x.array[:] += x[j] * func.state[idx].x.array
        for j, func in enumerate(standard_states):
            phi[idx].x.array[:] += x[len(active_set) + j] * func[idx].x.array
    sum_points = 0
    for i in range(len(active_set)):
        sum_points += x[i]
    return 0.5 * calculateL2InnerProduct(phi, phi, params) + sum_points
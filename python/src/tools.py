from dolfinx import fem, mesh, geometry
import numpy as np
from typing import List

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
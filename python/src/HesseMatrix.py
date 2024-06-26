import numpy as np
from ufl import ds, dx, grad, inner, dot
from typing import List, Tuple
from dolfinx import fem, mesh, plot, io, geometry
from src.solveStateEquation import solveStateEquation, getSourceTerm
from mpi4py import MPI

def calculateL2InnerProduct(firstState: List[fem.Function], secondState: List[fem.Function], params) -> np.float64:
    sum = 0
    for idx, __ in enumerate(firstState):
        energy_form = fem.form(inner(firstState[idx], secondState[idx]) * dx)
        energy_local = fem.assemble_scalar(energy_form)
        energy_global = params.V.mesh.comm.allreduce(energy_local, op=MPI.SUM)
        sum += energy_global
    return sum * params.dt * params.T

class HesseMatrix:
    def __init__(self, active_set, params) -> None:
        self.params = params
        self.standard_states = self.computeStandardEntries()
        self.active_set = []
        n = 2 * self.params.d
        self.matrix = np.zeros((n, n))
        for i, firstState in enumerate(self.standard_states):
            for j, secondState in enumerate(self.standard_states):
                self.matrix[i, j] = calculateL2InnerProduct(firstState, secondState, params)
        working_set = []
        for element in active_set:
            working_set.append(element)
            self.update(working_set)
     
    def computeStandardEntries(self):
        states = []
        signal_one = lambda t: t
        signal_zero = lambda t: 0
        signal_t = lambda t: 1
        g1 = getSourceTerm(self.params.x1, self.params)
        g2 = getSourceTerm(self.params.x2, self.params)
        state, __ = solveStateEquation([g1, g2], [signal_t, signal_zero], self.params)
        states.append(state)
        state, __ = solveStateEquation([g1, g2], [signal_zero, signal_t], self.params)
        states.append(state)
        state, __ = solveStateEquation([g1, g2], [signal_one, signal_zero], self.params)
        states.append(state)
        state, __ = solveStateEquation([g1, g2], [signal_zero, signal_one], self.params)
        states.append(state)
        return states
    
    #TODO: REFACTOR UPDATE FUNCTION
    def update(self, active_set):
        n = len(active_set) + 2 * self.params.d
        new_matrix = np.zeros((n, n))
        standard_block = self.matrix[-2 * self.params.d:, -2 * self.params.d:]
        new_matrix[-2 * self.params.d:, -2 * self.params.d:] = standard_block
        idx = 0
        for extremal in self.active_set:
            if extremal not in active_set:
                np.delete(self.matrix, idx, axis=0)
                np.delete(self.matrix, idx, axis=1)
            else:
                idx += 1
        new_row = np.ones(n)
        new_state = active_set[-1].state
        idx = 0
        for extremal in active_set:
            new_row[idx]  = calculateL2InnerProduct(new_state, extremal.state, self.params)
            idx += 1
        for state in self.standard_states:
            new_row[idx] = calculateL2InnerProduct(new_state, state, self.params)
        temp_row = np.zeros(n - 1)
        temp_row[:idx] = new_row[:idx]
        temp_row[idx:] = new_row[idx + 1:]
        print('new row:', new_row)
        print('temp row:', temp_row)
        temp_matrix = np.insert(self.matrix, idx, temp_row, axis=0)
        self.matrix = np.insert(temp_matrix, idx, new_row, axis=1)
        self.active_set = active_set[:]
            
        
        
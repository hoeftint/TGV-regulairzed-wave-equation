import numpy as np
from ufl import ds, dx, grad, inner, dot
from typing import List, Tuple
from dolfinx import fem, mesh, plot, io, geometry
from src.solveStateEquation import solveStateEquation, getSourceTerm, buildControlFunction
from src.helpers import calculateL2InnerProduct

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
            self.extendHesse(element)

    def computeStandardEntries(self):
        states = []
        signal_t = lambda t: t
        signal_zero = lambda t: 0
        signal_one = lambda t: 1
        g1 = getSourceTerm(self.params.x1, self.params)
        g2 = getSourceTerm(self.params.x2, self.params)
        control = buildControlFunction([g1, g2], [signal_t, signal_zero], self.params)
        state = solveStateEquation(control, self.params)
        states.append(state)
        control = buildControlFunction([g1, g2], [signal_zero, signal_t], self.params)
        state = solveStateEquation(control, self.params)
        states.append(state)
        control = buildControlFunction([g1, g2], [signal_one, signal_zero], self.params)
        state = solveStateEquation(control, self.params)
        states.append(state)
        control = buildControlFunction([g1, g2], [signal_zero, signal_one], self.params)
        state = solveStateEquation(control, self.params)
        states.append(state)
        return states

    # If there is a new point contained in the input argument, build a bigger matrix
    def extendMatrix(self, newPoint):
        idxNewPoint = len(self.active_set)
        newRow = np.zeros(len(self.active_set) + 1 + 2 * self.params.d)
        for idx, point in enumerate(self.active_set):
            newRow[idx] = calculateL2InnerProduct(newPoint.state, point.state, self.params)
        newRow[idxNewPoint] = calculateL2InnerProduct(newPoint.state, newPoint.state, self.params)
        for idx, state in enumerate(self.standard_states):
            newRow[idxNewPoint + 1 + idx] = calculateL2InnerProduct(newPoint.state, state, self.params)
        tempRow = np.delete(newRow, idxNewPoint)
        tempMatrix = np.insert(self.matrix, idxNewPoint, tempRow, axis=0)
        self.matrix = np.insert(tempMatrix, idxNewPoint, newRow, axis=1)
        self.active_set.append(newPoint)
        
    def pruneMatrix(self, active_set, weights):
        if (len(active_set) != len(weights)):
            raise Exception('The size of active_set is not matching the size of weights')
        idx = 0
        for point in self.active_set:
            if point not in active_set:
                self.matrix = np.delete(self.matrix, idx, axis=0)
                self.matrix = np.delete(self.matrix, idx, axis=1)
            else:
                idx = idx + 1
        self.active_set[:] = active_set

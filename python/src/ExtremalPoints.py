from enum import Enum
import numpy as np
from dolfinx import fem
from src.solveStateEquation import solveStateEquation, getSourceTerm

class ExtremalPoint:
    def __init__(self, sigma: np.ndarray, x_0: float, type: bool, params, idx: int) -> None:
        self.sigma = sigma
        self.x_0 = x_0
        self.type = type
        self.params = params
        self.idx = idx
        self.state = self.computeState()

    def value(self, x):
        if (self.type == 0):
            return (np.zeros_like(self.sigma) if x < self.x_0 else self.sigma/ self.params.alpha)
        else:
            if self.x_0 < self.params.T/2:
                return (self.sigma * (self.x_0 - x) / self.params.beta if x < self.x_0 else np.zeros_like(self.sigma))
            else:
                return (self.sigma * (x - self.x_0) / self.params.beta if x > self.x_0 else np.zeros_like(self.sigma))
            
    def computeState(self):
        s1 = lambda t: self.value(t)[0]
        s2 = lambda t: self.value(t)[1]
        g1 = getSourceTerm(self.params.x1, self.params)
        g2 = getSourceTerm(self.params.x2, self.params)
        state, __ = solveStateEquation([g1, g2], [s1, s2], self.params)
        return (state)
        

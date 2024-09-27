from enum import Enum
import numpy as np
from dolfinx import fem
from src.solutionOperators import solveStateEquation, getSourceTerm, buildControlFunction, solveAdjointEquation
from src.helpers import calculateL2InnerProduct

class ExtremalPoint:
	def __init__(self, sigma: np.ndarray, x_0: float, type: bool, params) -> None:
		self.sigma = sigma
		self.x_0 = x_0
		self.type = type
		self.params = params
		self.idx = int(x_0/params.dt)
		self.state = self.computeState()
		#self.adjoint = solveAdjointEquation(self.state, self.params)
		self.firstDual = self.computeFirstDualPart()
		#self.secondDual = self.computeSecondDualPart()
		if len(params.yd) > 0:
			self.standardInner = calculateL2InnerProduct(self.state, params.yd, params)

	def value(self, x):
		if (self.type == 0):
			return (np.zeros_like(self.sigma) if x < self.x_0 else self.sigma)
		else:
			if self.x_0 < self.params.T/2:
				return (self.sigma * (self.x_0 - x) if x < self.x_0 else np.zeros_like(self.sigma))
			else:
				return (self.sigma * (x - self.x_0) if x > self.x_0 else np.zeros_like(self.sigma))

	'''

	def value(self, x):
		if (self.type == 0):
			return (np.zeros_like(self.sigma) if x < self.x_0 else self.sigma/self.params.alpha)
		else:
			if self.x_0 < self.params.T/2:
				return (self.sigma * (self.x_0 - x) / self.params.beta if x < self.x_0 else np.zeros_like(self.sigma))
			else:
				return (self.sigma * (x - self.x_0) / self.params.beta if x > self.x_0 else np.zeros_like(self.sigma))
	'''

	def computeState(self):
		s1 = lambda t: self.value(t)[0]
		s2 = lambda t: self.value(t)[1]
		g1 = getSourceTerm(self.params.x1, self.params)
		g2 = getSourceTerm(self.params.x2, self.params)
		control = buildControlFunction([g1, g2], [s1, s2], self.params)
		if not self.params.useDummy:
			state = solveStateEquation(control, self.params)
		else:
			return control
		return state
	
	def computeFirstDualPart(self):
		if self.type == 0:
			primitive = lambda t: (0. if t < self.x_0 else t - self.x_0) #/ self.params.alpha
		elif self.type == 1 and self.x_0 < self.params.T/2:
			primitive = lambda t: (t * self.x_0 - 0.5 * t**2 if t < self.x_0 else 0.5 * self.x_0**2) #/ self.params.beta
		elif (self.type == 1 and self.x_0 >= self.params.T/2):
			primitive = lambda t: (0. if t < self.x_0 else 0.5 * t**2 - t * self.x_0) #/ self.params.beta
		s1 = lambda t: self.sigma[0] * primitive(t)
		s2 = lambda t: self.sigma[1] * primitive(t)
		g1 = getSourceTerm(self.params.x1, self.params)
		g2 = getSourceTerm(self.params.x2, self.params)
		control = buildControlFunction([g1, g2], [s1, s2], self.params)
		state = solveStateEquation(control, self.params)
		firstDual = solveAdjointEquation(state, self.params)
		return state
	
	def computeSecondDualPart(self):
		if self.type == 0:
			primitive = lambda t: (0. if t < self.x_0 else 0.5 * t**2 - self.x_0 * t + 0.5 * self.x_0**2) / self.params.alpha
		elif self.type == 1 and self.x_0 < self.params.T/2:
			primitive = lambda t: (0.5 * t**2 * self.x_0 - 1/6 * t**3 if t < self.x_0 else 0.5 * t * self.x_0**2 - 1/6 * t**3 ) / self.params.beta
		elif self.type == 1 and self.x_0 >= self.params.T/2:
			primitive = lambda t: (0. if t < self.x_0 else 1/6 * t**3 - 0.5 * t**2 * self.x_0 + 1/3 * self.x_0**3) / self.params.beta
		s1 = lambda t: self.sigma[0] * primitive(t)
		s2 = lambda t: self.sigma[1] * primitive(t)
		g1 = getSourceTerm(self.params.x1, self.params)
		g2 = getSourceTerm(self.params.x2, self.params)
		control = buildControlFunction([g1, g2], [s1, s2], self.params)
		state = solveStateEquation(control, self.params)
		secondDual = solveAdjointEquation(state, self.params)
		return state
from dataclasses import dataclass
import osqp
from scipy import sparse
import numpy as np
from src.HesseMatrix import calculateL2InnerProduct, HesseMatrix
from src.helpers import calculateL2InnerProduct
from src.ExtremalPoints import ExtremalPoint
from typing import List

def computeOSQPStep(weights, slope, y_shift, active_set: List[ExtremalPoint], hesse: HesseMatrix, params) -> tuple[np.array, np.array, np.array]:
    x = np.concatenate((weights, slope, y_shift))
    prob = osqp.OSQP()
    P = sparse.csr_matrix(hesse.matrix)
    standard_states = hesse.standard_states
    n = len(active_set) + 2 * params.d
    vectorStandardInner = np.zeros((n,))
    for idx, func in enumerate(active_set):
        vectorStandardInner[idx] = func.standardInner
    for idx, func in enumerate(standard_states):
        vectorStandardInner[len(active_set) + idx] = calculateL2InnerProduct(params.yd, standard_states[idx], params)
    additional_ones = np.zeros_like(x)
    for idx in range(len(active_set)):
        additional_ones[idx] = 1
    q = - vectorStandardInner + additional_ones
    l = np.zeros_like(x)
    for idx in range(len(active_set), n):
        l[idx] = -np.inf
    u = np.ones_like(x) * np.inf
    A = sparse.identity(n)
    prob.setup(P, q, A, l, u, alpha=1, verbose=False, eps_prim_inf=1e-10, eps_dual_inf=1e-10)
    #prob.warm_start(x=x)
    res = prob.solve()
    x = res.x
    return np.clip(x[:len(active_set)], a_min=0, a_max=None), x[-2*params.d:-params.d], x[-params.d:]
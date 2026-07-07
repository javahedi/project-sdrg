# utils.py

import numpy as np


def generate_positions(N, L):
    """Randomly place N spins on a chain of length L."""
    return np.sort(np.random.choice(L, size=N, replace=False))

def initial_couplings(positions, alpha):
    """Return dict {(i,j): J_ij}."""
    J = {}
    N = len(positions)
    for i in range(N):
        for j in range(i+1, N):
            dist = abs(positions[i] - positions[j])
            J[(i, j)] = dist ** (-alpha)
    return J

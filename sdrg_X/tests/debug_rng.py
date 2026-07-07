"""
Debug script to check if RNG behaves consistently between original and Numba.
"""

import numpy as np
from sdrgX_entropy import sample_pair_state
from sdrgX_entropy_numba import sample_pair_state_nb

# Test state sampling with same parameters
np.random.seed(100)
J = 1.0
T = 0.1

print("Testing state sampling with seed=100, J=1.0, T=0.1")
print("="*60)

# Original version
np.random.seed(100)
orig_states = [sample_pair_state(J, T) for _ in range(10)]
print(f"Original: {orig_states}")

# Numba version
np.random.seed(100)
numba_states = [sample_pair_state_nb(J, T) for _ in range(10)]
print(f"Numba:    {numba_states}")

if orig_states == numba_states:
    print("✓ States match!")
else:
    print("❌ States DON'T match!")
    mismatches = sum(1 for o, n in zip(orig_states, numba_states) if o != n)
    print(f"   Mismatches: {mismatches}/10")

# Check probabilities
print("\n" + "="*60)
print("Checking probability distribution...")
np.random.seed(200)
N_samples = 10000
orig_samples = np.array([sample_pair_state(J, T) for _ in range(N_samples)])

np.random.seed(200)
numba_samples = np.array([sample_pair_state_nb(J, T) for _ in range(N_samples)])

print(f"\nOriginal distribution (out of {N_samples}):")
for s in range(4):
    count = np.sum(orig_samples == s)
    print(f"  State {s}: {count} ({count/N_samples*100:.1f}%)")

print(f"\nNumba distribution (out of {N_samples}):")
for s in range(4):
    count = np.sum(numba_samples == s)
    print(f"  State {s}: {count} ({count/N_samples*100:.1f}%)")

# Expected probabilities
beta = 1.0 / T
E = np.array([-J/2, 0, 0, J/2])
w = np.exp(-beta * E)
p = w / np.sum(w)

print(f"\nExpected distribution:")
for s in range(4):
    print(f"  State {s}: {p[s]*100:.1f}%")

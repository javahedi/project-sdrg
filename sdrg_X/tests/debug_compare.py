"""
Debug script to compare pair-by-pair outputs from original and Numba versions.
"""

import numpy as np
from sdrgX_entropy import sdrg_pairing_finite_T, entanglement_entropy_finite_T
from sdrgX_entropy_numba import sdrg_pairing_numba, entanglement_entropy_numba
from utils import generate_positions, initial_couplings

# Set seed for reproducibility
np.random.seed(42)

N = 10
L = 50
alpha = 2.0
T = 0.0  # No thermal fluctuations for debugging

print("Generating test case...")
positions = generate_positions(N, L)
print(f"Positions: {positions}")

J_dict = initial_couplings(positions, alpha)
print(f"\nCouplings (dict): {len(J_dict)} pairs")

# Convert to array for Numba version
J_array = np.zeros((N, N), dtype=np.float64)
for (i, j), val in J_dict.items():
    J_array[i, j] = val

print("\n" + "="*70)
print("ORIGINAL VERSION")
print("="*70)
pairs_orig = sdrg_pairing_finite_T(positions, J_dict, T)
print(f"Number of pairs: {len(pairs_orig)}")
print("\nFirst 5 pairs (r1, r2, s):")
for i, (r1, r2, s) in enumerate(pairs_orig[:5]):
    print(f"  {i}: ({r1}, {r2}, s={s})")

S_orig = entanglement_entropy_finite_T(pairs_orig, L)
print(f"\nEntropy at position 25: {S_orig[25]:.4f}")
print(f"Max entropy: {np.max(S_orig):.4f}")

print("\n" + "="*70)
print("NUMBA VERSION")
print("="*70)
pairs_r1, pairs_r2, pairs_s, n_pairs = sdrg_pairing_numba(positions, J_array, T, N)
print(f"Number of pairs: {n_pairs}")
print("\nFirst 5 pairs (r1, r2, s):")
for i in range(min(5, n_pairs)):
    print(f"  {i}: ({pairs_r1[i]}, {pairs_r2[i]}, s={pairs_s[i]})")

S_numba = entanglement_entropy_numba(pairs_r1, pairs_r2, pairs_s, n_pairs, L)
print(f"\nEntropy at position 25: {S_numba[25]:.4f}")
print(f"Max entropy: {np.max(S_numba):.4f}")

print("\n" + "="*70)
print("COMPARISON")
print("="*70)

# Compare pairs
print(f"Number of pairs - Original: {len(pairs_orig)}, Numba: {n_pairs}")

if len(pairs_orig) == n_pairs:
    print("\nPair-by-pair comparison:")
    mismatches = 0
    for i in range(n_pairs):
        r1_o, r2_o, s_o = pairs_orig[i]
        r1_n, r2_n, s_n = pairs_r1[i], pairs_r2[i], pairs_s[i]

        match = (r1_o == r1_n and r2_o == r2_n and s_o == s_n)
        if not match:
            print(f"  Pair {i}: Orig=({r1_o},{r2_o},s={s_o}) vs Numba=({r1_n},{r2_n},s={s_n}) ❌")
            mismatches += 1

    if mismatches == 0:
        print("  All pairs match! ✓")
    else:
        print(f"\n  Total mismatches: {mismatches}/{n_pairs}")
else:
    print("  Different number of pairs! ❌")

# Compare entropy
S_diff = np.abs(S_orig - S_numba)
print(f"\nEntropy comparison:")
print(f"  Max absolute difference: {np.max(S_diff):.6f}")
print(f"  Mean absolute difference: {np.mean(S_diff):.6f}")
print(f"  Positions with differences: {np.sum(S_diff > 1e-10)}/{L}")

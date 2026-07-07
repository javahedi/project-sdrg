"""
Benchmark with larger, more realistic parameters to measure true speedup.
"""

import numpy as np
import time
from sdrgX_entropy import run_sdrg_entropy_multi_T
from sdrgX_entropy_numba import run_sdrg_entropy_multi_T_numba

print("="*70)
print("LARGE-SCALE BENCHMARK")
print("="*70)

# Realistic parameters (but scaled down from full 500 disorder realizations)
N = 100
L = 1000
alpha = 3.0
T_list = [0.0, 0.1]
n_disorder = 20  # Scaled down from 500 for faster testing
n_thermal = 50   # Scaled down from 100

print(f"\nParameters:")
print(f"  N={N}, L={L}, alpha={alpha}")
print(f"  T_list={T_list}")
print(f"  n_disorder={n_disorder}, n_thermal={n_thermal}")
print(f"  Total SDRG iterations: {n_disorder * len(T_list) * n_thermal} (~2000)")
print()

# Run original version
print("-"*70)
print("Running ORIGINAL version...")
print("-"*70)
np.random.seed(999)
t0 = time.time()
run_sdrg_entropy_multi_T(
    N=N,
    L=L,
    alpha=alpha,
    T_list=T_list,
    n_disorder=n_disorder,
    n_thermal=n_thermal,
    outdir="benchmark_original"
)
t_orig = time.time() - t0
print(f"Original: {t_orig:.2f} seconds")

# Run Numba version
print()
print("-"*70)
print("Running NUMBA version...")
print("-"*70)
np.random.seed(999)
t0 = time.time()
run_sdrg_entropy_multi_T_numba(
    N=N,
    L=L,
    alpha=alpha,
    T_list=T_list,
    n_disorder=n_disorder,
    n_thermal=n_thermal,
    outdir="benchmark_numba"
)
t_numba = time.time() - t0
print(f"Numba: {t_numba:.2f} seconds")

# Results
print()
print("="*70)
print("BENCHMARK RESULTS")
print("="*70)
print(f"Original version: {t_orig:>10.2f} seconds")
print(f"Numba version:    {t_numba:>10.2f} seconds")
speedup = t_orig / t_numba if t_numba > 0 else 0
print(f"Speedup:          {speedup:>10.2f}x")
print()

if speedup >= 20:
    print(f"✓✓✓ EXCELLENT: {speedup:.1f}x speedup achieved!")
elif speedup >= 10:
    print(f"✓✓ GOOD: {speedup:.1f}x speedup achieved")
elif speedup >= 5:
    print(f"✓ MODERATE: {speedup:.1f}x speedup achieved")
else:
    print(f"⚠ LOW: Only {speedup:.1f}x speedup (target: 20x+)")

print()
print(f"Estimated time for full simulation (n_disorder=500, n_thermal=100):")
print(f"  Original: {t_orig * (500/n_disorder) * (100/n_thermal) / 60:.1f} minutes")
print(f"  Numba:    {t_numba * (500/n_disorder) * (100/n_thermal) / 60:.1f} minutes")
print("="*70)

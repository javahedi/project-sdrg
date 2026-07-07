#!/usr/bin/env python3
"""
Verification script for parallel SDRG-X implementation.

This script runs both serial and parallel versions with small
parameters to verify correctness and measure speedup.
"""

import time
import json
import os
import shutil
from sdrgX_entropy import run_sdrg_entropy_multi_T
from sdrgX_entropy_parallel import run_sdrg_entropy_multi_T_parallel


def load_json(filepath):
    """Load and return JSON data."""
    with open(filepath, 'r') as f:
        return json.load(f)


def compare_results(serial_data, parallel_data, tolerance=1e-2):
    """
    Compare serial and parallel results.

    Returns (is_similar, max_difference)
    """
    max_diff = 0.0

    for T in serial_data['T_list']:
        T_str = str(float(T))

        serial_S = serial_data['S_l_by_T'][T_str]
        parallel_S = parallel_data['S_l_by_T'][T_str]

        # Compute maximum absolute difference
        diff = max(abs(s - p) for s, p in zip(serial_S, parallel_S))
        max_diff = max(max_diff, diff)

    is_similar = max_diff < tolerance
    return is_similar, max_diff


def test_small_scale():
    """Test with small parameters for quick verification."""
    print("=" * 60)
    print("SMALL-SCALE VERIFICATION TEST")
    print("=" * 60)

    # Test parameters (small for quick testing)
    params = {
        'N': 50,
        'L': 500,
        'alpha': 3.0,
        'T_list': [0.0, 0.1, 0.5],
        'n_disorder': 10,
        'n_thermal': 20
    }

    print("\nTest parameters:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print()

    # Run serial version
    print("Running SERIAL version...")
    serial_dir = "verify_serial"
    if os.path.exists(serial_dir):
        shutil.rmtree(serial_dir)

    t0_serial = time.time()
    run_sdrg_entropy_multi_T(
        **params,
        outdir=serial_dir
    )
    t_serial = time.time() - t0_serial

    print(f"Serial time: {t_serial:.2f} seconds\n")

    # Run parallel version
    print("Running PARALLEL version...")
    parallel_dir = "verify_parallel"
    if os.path.exists(parallel_dir):
        shutil.rmtree(parallel_dir)

    t0_parallel = time.time()
    run_sdrg_entropy_multi_T_parallel(
        **params,
        n_workers=4,
        base_seed=42,
        outdir=parallel_dir
    )
    t_parallel = time.time() - t0_parallel

    print(f"Parallel time: {t_parallel:.2f} seconds\n")

    # Compare results
    print("Comparing results...")
    serial_data = load_json(os.path.join(serial_dir, "S_l_all_T.json"))
    parallel_data = load_json(os.path.join(parallel_dir, "S_l_all_T.json"))

    is_similar, max_diff = compare_results(serial_data, parallel_data, tolerance=0.5)

    print(f"  Maximum difference: {max_diff:.6f}")
    print(f"  Results similar: {is_similar}")

    if t_serial > 0:
        speedup = t_serial / t_parallel
        print(f"  Speedup: {speedup:.2f}x")

    # Cleanup
    print("\nCleaning up verification directories...")
    shutil.rmtree(serial_dir)
    shutil.rmtree(parallel_dir)

    return is_similar


def test_reproducibility():
    """Test that parallel version produces reproducible results."""
    print("\n" + "=" * 60)
    print("REPRODUCIBILITY TEST")
    print("=" * 60)

    params = {
        'N': 50,
        'L': 500,
        'alpha': 3.0,
        'T_list': [0.0, 0.5],
        'n_disorder': 5,
        'n_thermal': 10,
        'n_workers': 2,
        'base_seed': 12345
    }

    print("\nRunning parallel version twice with same seed...")

    # First run
    run1_dir = "verify_run1"
    if os.path.exists(run1_dir):
        shutil.rmtree(run1_dir)

    run_sdrg_entropy_multi_T_parallel(**params, outdir=run1_dir)
    data1 = load_json(os.path.join(run1_dir, "S_l_all_T.json"))

    # Second run
    run2_dir = "verify_run2"
    if os.path.exists(run2_dir):
        shutil.rmtree(run2_dir)

    run_sdrg_entropy_multi_T_parallel(**params, outdir=run2_dir)
    data2 = load_json(os.path.join(run2_dir, "S_l_all_T.json"))

    # Compare
    is_identical, max_diff = compare_results(data1, data2, tolerance=1e-10)

    print(f"\n  Maximum difference: {max_diff:.6f}")
    print(f"  Results identical: {is_identical}")

    # Cleanup
    shutil.rmtree(run1_dir)
    shutil.rmtree(run2_dir)

    return is_identical


def test_performance_scaling():
    """Test speedup with different worker counts."""
    print("\n" + "=" * 60)
    print("PERFORMANCE SCALING TEST")
    print("=" * 60)

    params = {
        'N': 50,
        'L': 500,
        'alpha': 3.0,
        'T_list': [0.0, 0.5],
        'n_disorder': 50,
        'n_thermal': 20
    }

    worker_counts = [1, 2, 4, 8]
    times = {}

    for n_workers in worker_counts:
        print(f"\nTesting with {n_workers} worker(s)...")
        test_dir = f"verify_workers_{n_workers}"
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

        t0 = time.time()
        run_sdrg_entropy_multi_T_parallel(
            **params,
            n_workers=n_workers,
            base_seed=42,
            outdir=test_dir
        )
        elapsed = time.time() - t0
        times[n_workers] = elapsed

        print(f"  Time: {elapsed:.2f} seconds")

        shutil.rmtree(test_dir)

    # Print scaling summary
    print("\n" + "-" * 40)
    print("SCALING SUMMARY")
    print("-" * 40)
    baseline = times[1]
    print(f"{'Workers':<10} {'Time (s)':<12} {'Speedup':<10}")
    print("-" * 40)
    for n_workers in worker_counts:
        t = times[n_workers]
        speedup = baseline / t
        print(f"{n_workers:<10} {t:<12.2f} {speedup:<10.2f}x")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("SDRG-X PARALLEL IMPLEMENTATION VERIFICATION")
    print("*" * 60)

    # Run tests
    test1_passed = test_small_scale()
    test2_passed = test_reproducibility()

    # Optional performance test (can be slow)
    print("\n" + "=" * 60)
    import sys
    # Only prompt if running interactively (with a terminal)
    if sys.stdin.isatty():
        response = input("Run performance scaling test? (slower, ~2-5 min) [y/N]: ")
        if response.lower() == 'y':
            test_performance_scaling()
    else:
        print("Skipping performance scaling test (non-interactive mode)")

    # Final summary
    print("\n" + "*" * 60)
    print("VERIFICATION SUMMARY")
    print("*" * 60)
    print(f"  Small-scale test: {'PASSED' if test1_passed else 'FAILED'}")
    print(f"  Reproducibility test: {'PASSED' if test2_passed else 'FAILED'}")

    if test1_passed and test2_passed:
        print("\n✓ All tests passed! Parallel implementation is verified.")
    else:
        print("\n✗ Some tests failed. Please investigate.")

    print("\n")

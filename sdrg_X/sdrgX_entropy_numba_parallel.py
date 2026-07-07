import numpy as np
import json
import os
import multiprocessing as mp
from numba import jit


# ============================================================
# Import Numba-optimized functions
# ============================================================

@jit(nopython=True)
def generate_positions_nb(N, L):
    """
    Numba-compiled version of generate_positions.
    Randomly place N spins on a chain of length L using Fisher-Yates shuffle.
    """
    # Create array of all possible positions
    all_pos = np.arange(L, dtype=np.int64)

    # Fisher-Yates shuffle to get first N elements
    for i in range(N):
        # Pick random index from i to L-1
        j = i + np.random.randint(0, L - i)
        # Swap
        all_pos[i], all_pos[j] = all_pos[j], all_pos[i]

    # Take first N positions and sort
    positions = all_pos[:N].copy()
    positions.sort()
    return positions


@jit(nopython=True)
def initial_couplings_nb(positions, alpha):
    """
    Numba-compiled version of initial_couplings.
    Returns 2D array J[i,j] for i < j.
    """
    N = len(positions)
    J = np.zeros((N, N), dtype=np.float64)

    alpha = float(alpha)  # Ensure alpha is float

    for i in range(N):
        for j in range(i + 1, N):
            dist = float(abs(positions[i] - positions[j]))
            J[i, j] = dist ** (-alpha)

    return J


@jit(nopython=True)
def sample_pair_state_nb(J_val, T):
    """
    Sample one of the four eigenstates s = 0,1,2,3
    for a bond of strength J at temperature T.
    Numba-compatible version using manual cumulative sum.
    """
    if T == 0.0:
        return 0  # singlet only

    beta = 1.0 / T

    # Energies for the 4 states
    # s=0: -J/2 (singlet, entangled)
    # s=1,2: 0 (triplet, not entangled)
    # s=3: J/2 (triplet, entangled)
    E0 = -J_val / 2.0
    E1 = 0.0
    E2 = 0.0
    E3 = J_val / 2.0

    # Boltzmann weights
    w0 = np.exp(-beta * E0)
    w1 = np.exp(-beta * E1)
    w2 = np.exp(-beta * E2)
    w3 = np.exp(-beta * E3)

    Z = w0 + w1 + w2 + w3

    # Probabilities
    p0 = w0 / Z
    p1 = w1 / Z
    p2 = w2 / Z
    p3 = w3 / Z

    # Manual cumulative sum for sampling
    cum_probs = np.array([p0, p0 + p1, p0 + p1 + p2, 1.0])
    r = np.random.random()

    # Find which bin the random number falls into
    if r < cum_probs[0]:
        return 0
    elif r < cum_probs[1]:
        return 1
    elif r < cum_probs[2]:
        return 2
    else:
        return 3


@jit(nopython=True)
def sdrg_pairing_numba(positions, J_init, T, N, heuristic='strongest'):
    """
    Perform SDRG-X pairing and return arrays of pairs.

    Parameters:
    -----------
    positions : array of int64, shape (N,)
        Positions of spins on the chain
    J_init : array of float64, shape (N, N)
        Initial coupling matrix (upper triangular)
    T : float
        Temperature
    N : int
        Number of sites
    heuristic : str
        Pairing heuristic ('strongest', 'random' supported)

    Returns:
    --------
    pairs_r1 : array of int64
        First position of each pair
    pairs_r2 : array of int64
        Second position of each pair
    pairs_s : array of int64
        Eigenstate for each pair (0,1,2,3)
    n_pairs : int
        Actual number of pairs formed
    """
    # Make a copy of J to modify
    J = J_init.copy()

    # Pre-allocate output arrays (max N//2 pairs)
    max_pairs = N // 2
    pairs_r1 = np.zeros(max_pairs, dtype=np.int64)
    pairs_r2 = np.zeros(max_pairs, dtype=np.int64)
    pairs_s = np.zeros(max_pairs, dtype=np.int64)

    n_pairs = 0
    n_remaining = N

    # Continue until we can't form more pairs
    while n_remaining > 1 and n_pairs < max_pairs:
        # Find maximum coupling in upper triangle
        max_J = -1.0
        max_i = -1
        max_j = -1

        if heuristic == 'strongest':
            idx = np.argmax(J)
        elif heuristic == 'random':
            upper = np.triu(J, k=1)
            upper_flat = upper.flatten()
            total = np.sum(upper_flat)
            if total > 0:
                probs = upper_flat / total
                cum_probs = np.cumsum(probs)
                r = np.random.random()
                idx = np.searchsorted(cum_probs, r)
            else:
                idx = 0
        else:
            # Default to strongest for unknown heuristics
            idx = np.argmax(J)

        max_i = idx // N
        max_j = idx % N
        max_J = J[max_i, max_j]

        # If no valid coupling found, break
        if max_J <= 0.0:
            break

        # Sample eigenstate for this pair
        s = sample_pair_state_nb(max_J, T)

        # Record the pair (using actual positions, not indices)
        pairs_r1[n_pairs] = positions[max_i]
        pairs_r2[n_pairs] = positions[max_j]
        pairs_s[n_pairs] = s
        n_pairs += 1

        # Remove this pair by zeroing out rows and columns
        J[max_i, :] = 0.0
        J[:, max_i] = 0.0
        J[max_j, :] = 0.0
        J[:, max_j] = 0.0

        n_remaining -= 2

    return pairs_r1, pairs_r2, pairs_s, n_pairs


@jit(nopython=True)
def entanglement_entropy_numba(pairs_r1, pairs_r2, pairs_s, n_pairs, L):
    """
    Compute entanglement entropy S(l) from pairs.

    Only s=0 (singlet) and s=3 (entangled triplet) contribute.

    Parameters:
    -----------
    pairs_r1, pairs_r2 : arrays of int64
        Positions of paired spins
    pairs_s : array of int64
        Eigenstates (0,1,2,3)
    n_pairs : int
        Actual number of pairs
    L : int
        Chain length

    Returns:
    --------
    S : array of float64, shape (L,)
        Entanglement entropy at each position
    """
    S = np.zeros(L, dtype=np.float64)
    ln2 = np.log(2.0)

    for l in range(L):
        crossings = 0
        for k in range(n_pairs):
            s = pairs_s[k]
            # Only entangled states contribute
            if s != 0 and s != 3:
                continue

            r1 = pairs_r1[k]
            r2 = pairs_r2[k]

            # Check if pair crosses the cut at position l
            if (r1 < l < r2) or (r2 < l < r1):
                crossings += 1

        S[l] = ln2 * crossings

    return S


# ============================================================
# Worker function for multiprocessing
# ============================================================

def _worker_wrapper_numba(args):
    """
    Wrapper to unpack arguments for multiprocessing.
    Needed because lambda functions cannot be pickled.
    """
    return process_disorder_realization_numba(*args)


def process_disorder_realization_numba(disorder_idx, N, L, alpha, T_list, n_thermal, base_seed=42):
    """
    Process one disorder realization using Numba-optimized functions.

    This function combines the parallelization strategy from sdrgX_entropy_parallel.py
    with the Numba-optimized functions from sdrgX_entropy_numba.py.

    Args:
        disorder_idx: Index of this disorder realization
        N: Number of spins
        L: Chain length
        alpha: Power-law exponent
        T_list: List of temperatures to process
        n_thermal: Number of thermal samples per temperature
        base_seed: Base seed for reproducibility

    Returns:
        (disorder_idx, {T: S_avg_array})
    """
    try:
        # Seed RNG based on disorder index for reproducibility
        # This ensures each disorder realization uses the same random sequence
        # regardless of which worker processes it
        np.random.seed(base_seed + disorder_idx)

        # Generate using Numba functions
        positions = generate_positions_nb(N, L)
        J_init = initial_couplings_nb(positions, alpha)

        # Sanitize J to prevent numerical issues
        J_init = np.nan_to_num(J_init, posinf=1e10, neginf=-1e10, nan=0.0)

        results_by_T = {}

        for T in T_list:
            T = float(T)
            S_thermal = []

            # For T=0, only one sample needed (deterministic)
            n_samples = n_thermal if T > 0 else 1

            for _ in range(n_samples):
                # Numba-optimized pairing
                pairs_r1, pairs_r2, pairs_s, n_pairs = sdrg_pairing_numba(
                    positions, J_init, T, N
                )

                # Numba-optimized entropy
                S = entanglement_entropy_numba(
                    pairs_r1, pairs_r2, pairs_s, n_pairs, L
                )
                S_thermal.append(S)

            # Average over thermal sampling
            results_by_T[float(T)] = np.mean(S_thermal, axis=0)

        return (disorder_idx, results_by_T)

    except Exception as e:
        print(f"\nError in disorder realization {disorder_idx}: {e}")
        return (disorder_idx, None)


# ============================================================
# Main parallelized driver with Numba optimization
# ============================================================

def run_sdrg_entropy_multi_T_numba_parallel(
    N=100,
    L=1000,
    alpha=2.0,
    T_list=(0.0, 0.1, 0.2, 0.5),
    n_disorder=1000,
    n_thermal=100,
    n_workers=None,       # defaults to cpu_count() - 1
    chunk_size=None,      # auto-calculated if None
    base_seed=42,         # for reproducibility
    outdir="sdrgX_data_numba_parallel"
):
    """
    Parallelized SDRG-X entropy calculation with Numba optimization.

    This function combines:
    - Numba JIT compilation (70-300x speedup over original)
    - Multiprocessing parallelization (6-8x speedup on 8 cores)
    - Expected combined speedup: 420-2400x over original implementation

    Args:
        N: Number of spins
        L: Chain length
        alpha: Power-law exponent
        T_list: List of temperatures
        n_disorder: Number of disorder realizations
        n_thermal: Number of thermal samples per temperature (T>0)
        n_workers: Number of parallel workers (defaults to cpu_count()-1)
        chunk_size: Tasks per worker batch (auto-calculated if None)
        base_seed: Base random seed for reproducibility
        outdir: Output directory for results
    """
    os.makedirs(outdir, exist_ok=True)

    # CRITICAL: Convert alpha to float to avoid integer exponentiation error
    alpha = float(alpha)

    # Auto-configure parallelization
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if chunk_size is None:
        chunk_size = max(1, n_disorder // (n_workers * 4))

    print(f"Configuration:")
    print(f"  N={N}, L={L}, alpha={alpha}")
    print(f"  T_list={list(T_list)}")
    print(f"  n_disorder={n_disorder}, n_thermal={n_thermal}")
    print(f"  n_workers={n_workers}, chunk_size={chunk_size}")
    print(f"  base_seed={base_seed}")
    print(f"  Optimization: Numba + Multiprocessing")
    print()

    # Prepare task arguments
    tasks = [
        (d, N, L, alpha, T_list, n_thermal, base_seed)
        for d in range(n_disorder)
    ]

    # Storage for results
    S_by_T = {float(T): [] for T in T_list}

    # Parallel execution with progress tracking
    print(f"Running {n_disorder} disorder realizations on {n_workers} workers...")

    with mp.Pool(n_workers) as pool:
        results = []
        for result in pool.imap_unordered(_worker_wrapper_numba, tasks, chunksize=chunk_size):
            results.append(result)
            print(f"\rCompleted {len(results)}/{n_disorder}", end="", flush=True)
        print()  # newline after progress

    # Filter out failed computations
    failed_count = sum(1 for _, res in results if res is None)
    if failed_count > 0:
        print(f"Warning: {failed_count} disorder realizations failed")
        results = [r for r in results if r[1] is not None]

    # Sort results by disorder_idx and aggregate
    results.sort(key=lambda x: x[0])

    for disorder_idx, results_by_T in results:
        for T in T_list:
            S_by_T[float(T)].append(results_by_T[float(T)])

    # Disorder average and save (identical to other versions)
    final_results = {}

    for T in T_list:
        T = float(T)
        S_avg = np.mean(S_by_T[T], axis=0)

        final_results[T] = S_avg.tolist()

        # Save individual temperature file
        data_T = {
            "N": N,
            "L": L,
            "alpha": alpha,
            "T": T,
            "n_disorder": len(S_by_T[T]),  # actual count (in case of failures)
            "n_thermal": n_thermal,
            "S_l": S_avg.tolist()
        }

        fname = f"S_l_T_{T:.3f}.json"
        with open(os.path.join(outdir, fname), "w") as f:
            json.dump(data_T, f, indent=2)

    # Save combined file
    combined = {
        "N": N,
        "L": L,
        "alpha": alpha,
        "T_list": list(map(float, T_list)),
        "n_disorder": len(results),  # actual count
        "n_thermal": n_thermal,
        "S_l_by_T": final_results
    }

    with open(os.path.join(outdir, "S_l_all_T.json"), "w") as f:
        json.dump(combined, f, indent=2)

    print(f"Saved entanglement entropy for all temperatures to {outdir}")


# ============================================================
# Run (with multiprocessing guard)
# ============================================================

if __name__ == "__main__":
    # Required for Windows compatibility and proper multiprocessing
    run_sdrg_entropy_multi_T_numba_parallel(
        N=100,
        L=1000,
        alpha=3.0,
        T_list=[0.0, 0.005, 0.01, 0.1, 1.0],
        n_disorder=500,
        n_thermal=100,
        n_workers=None,    # Auto-detect
        chunk_size=None,   # Auto-calculate
        base_seed=42       # For reproducibility
    )

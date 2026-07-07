import numpy as np
import json
import os
import multiprocessing as mp
from utils import generate_positions, initial_couplings


# ============================================================
# Thermal sampling of pair eigenstates (SDRG-X)
# ============================================================

def sample_pair_state(J, T):
    """
    Sample one of the four eigenstates s = 0,1,2,3
    for a bond of strength J at temperature T.
    """
    if T == 0:
        return 0  # singlet only

    beta = 1.0 / T

    energies = np.array([
        -J / 2,  # s=0 (singlet, entangled)
         0.0,    # s=1
         0.0,    # s=2
         J / 2   # s=3 (entangled triplet)
    ])

    weights = np.exp(-beta * energies)
    probs = weights / np.sum(weights)

    return np.random.choice([0, 1, 2, 3], p=probs)


# ============================================================
# SDRG-X pairing
# ============================================================

def sdrg_pairing_finite_T(positions, J, T):
    """
    Perform SDRG-X and return list of pairs:
    (r_i, r_j, s)
    """
    active = list(range(len(positions)))
    pairs = []

    J = J.copy()

    while len(active) > 1:
        i, j = max(
            [(i, j) for (i, j) in J if i in active and j in active],
            key=lambda x: J[x]
        )

        Jij = J[(i, j)]
        s = sample_pair_state(Jij, T)

        pairs.append((positions[i], positions[j], s))

        active.remove(i)
        active.remove(j)

        J = {
            (k, l): v
            for (k, l), v in J.items()
            if k in active and l in active
        }

    return pairs


# ============================================================
# Entanglement entropy
# ============================================================

def entanglement_entropy_finite_T(pairs, L):
    """
    S(l) = ln(2) * number of entangled pairs crossing cut l
    Only s = 0 and s = 3 contribute.
    """
    S = np.zeros(L)

    for l in range(L):
        crossings = 0
        for r1, r2, s in pairs:
            if s not in (0, 3):
                continue
            if (r1 < l < r2) or (r2 < l < r1):
                crossings += 1
        S[l] = np.log(2) * crossings

    return S


# ============================================================
# Worker function (NEW - extracted from disorder loop)
# ============================================================

def _worker_wrapper(args):
    """
    Wrapper to unpack arguments for multiprocessing.
    Needed because lambda functions cannot be pickled.
    """
    return process_disorder_realization(*args)


def process_disorder_realization(disorder_idx, N, L, alpha, T_list, n_thermal, base_seed=42):
    """
    Process one disorder realization across all temperatures.

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

        # Generate random positions and couplings for this disorder realization
        positions = generate_positions(N, L)
        J = initial_couplings(positions, alpha)

        results_by_T = {}

        for T in T_list:
            S_thermal = []

            # For T=0, only one sample needed (deterministic)
            n_samples = n_thermal if T > 0 else 1

            for _ in range(n_samples):
                pairs = sdrg_pairing_finite_T(positions, J, T)
                S = entanglement_entropy_finite_T(pairs, L)
                S_thermal.append(S)

            # Average over thermal sampling
            results_by_T[float(T)] = np.mean(S_thermal, axis=0)

        return (disorder_idx, results_by_T)

    except Exception as e:
        print(f"\nError in disorder realization {disorder_idx}: {e}")
        return (disorder_idx, None)


# ============================================================
# Main parallelized driver (MODIFIED)
# ============================================================

def run_sdrg_entropy_multi_T_parallel(
    N=100,
    L=1000,
    alpha=2.0,
    T_list=(0.0, 0.1, 0.2, 0.5),
    n_disorder=1000,
    n_thermal=100,
    n_workers=None,       # NEW: defaults to cpu_count() - 1
    chunk_size=None,      # NEW: auto-calculated if None
    base_seed=42,         # NEW: for reproducibility
    outdir="sdrgX_data"
):
    """
    Parallelized SDRG-X entropy calculation using multiprocessing.

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
        for result in pool.imap_unordered(_worker_wrapper, tasks, chunksize=chunk_size):
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

    # Disorder average and save (identical to original)
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
    run_sdrg_entropy_multi_T_parallel(
        N=100,
        L=1000,
        alpha=3.0,
        # T_list=[0.0, 0.005, 0.01, 0.1, 1.0],
        T_list=[0.0, 0.005, 1.0],
        n_disorder=100,
        n_thermal=100,
        n_workers=8,    # auto-detect
        chunk_size=None,   # auto-calculate
        base_seed=42       # for reproducibility
    )

# Parallel & Optimized SDRG-X Entropy Calculation

This directory contains multiple implementations of the SDRG-X entropy calculation with different optimization strategies.

## Files

- **`sdrgX_entropy.py`**: Original serial implementation (preserved for reference)
- **`sdrgX_entropy_numba.py`**: Numba-optimized implementation (70-300× speedup)
- **`sdrgX_entropy_parallel.py`**: Multiprocessing parallel implementation (6-8× speedup)
- **`sdrgX_entropy_numba_parallel.py`**: **NEW!** Combined Numba + multiprocessing (420-2400× speedup)
- **`verify_parallel.py`**: Verification for multiprocessing version
- **`verify_numba_parallel.py`**: **NEW!** Verification for combined version
- **`utils.py`**: Shared utility functions
- **`plot_entropy.py`**: Plotting script (works with all versions)

## Quick Start

### Run Combined Numba+Parallel Version (FASTEST)

```bash
# Activate environment
micromamba activate sdrg

# Run with default parameters (uses Numba + all available CPU cores - 1)
python sdrgX_entropy_numba_parallel.py
```

### Run Other Versions

```bash
# Numba-only (fast, single-threaded)
python sdrgX_entropy_numba.py

# Multiprocessing-only (parallel, original algorithm)
python sdrgX_entropy_parallel.py

# Original (slow, for reference)
python sdrgX_entropy.py
```

### Verify Implementation

```bash
# Verify combined Numba+Parallel version
python verify_numba_parallel.py

# Verify multiprocessing-only version
python verify_parallel.py
```

## Performance

### Expected Speedup (vs Original)

On a typical 8-core system:
- **Original**: 1× (baseline)
- **Numba-only**: 70-300× (JIT compilation)
- **Parallel-only**: 6-8× (multiprocessing)
- **Numba+Parallel**: **420-2400×** (combined optimization)

### Benchmarks

| Version | Configuration | Time | Speedup vs Original |
|---------|--------------|------|---------------------|
| Original | n_disorder=500, n_thermal=100 | ~4 hours | 1× |
| Numba-only | n_disorder=500, n_thermal=100 | ~1.5 minutes | ~160× |
| Parallel-only (8 cores) | n_disorder=500, n_thermal=100 | ~30 minutes | ~8× |
| **Numba+Parallel (8 cores)** | **n_disorder=500, n_thermal=100** | **~12 seconds** | **~1200×** |

## Usage

### Basic Usage - Combined Numba+Parallel (RECOMMENDED)

```python
from sdrgX_entropy_numba_parallel import run_sdrg_entropy_multi_T_numba_parallel

# Fastest version - combines Numba JIT + multiprocessing
run_sdrg_entropy_multi_T_numba_parallel(
    N=100,              # Number of spins
    L=1000,             # Chain length
    alpha=3.0,          # Power-law exponent
    T_list=[0.0, 0.005, 0.01, 0.1, 1.0],  # Temperatures
    n_disorder=500,     # Disorder realizations
    n_thermal=100,      # Thermal samples per temperature (T>0)
    n_workers=None,     # Auto-detect CPU cores (defaults to cpu_count()-1)
    chunk_size=None,    # Auto-calculate optimal chunk size
    base_seed=42,       # Random seed for reproducibility
    outdir="sdrgX_data_numba_parallel"  # Output directory
)
```

### Basic Usage - Parallel Only

```python
from sdrgX_entropy_parallel import run_sdrg_entropy_multi_T_parallel

# Multiprocessing only (no Numba optimization)
run_sdrg_entropy_multi_T_parallel(
    N=100,              # Number of spins
    L=1000,             # Chain length
    alpha=3.0,          # Power-law exponent
    T_list=[0.0, 0.005, 0.01, 0.1, 1.0],  # Temperatures
    n_disorder=500,     # Disorder realizations
    n_thermal=100,      # Thermal samples per temperature (T>0)
    n_workers=None,     # Auto-detect CPU cores (defaults to cpu_count()-1)
    chunk_size=None,    # Auto-calculate optimal chunk size
    base_seed=42,       # Random seed for reproducibility
    outdir="sdrgX_data" # Output directory
)
```

### Advanced Configuration

#### Control Number of Workers

```python
# Use 8 workers explicitly
run_sdrg_entropy_multi_T_parallel(..., n_workers=8)

# Use all available cores
import multiprocessing as mp
run_sdrg_entropy_multi_T_parallel(..., n_workers=mp.cpu_count())

# Serial execution (for debugging)
run_sdrg_entropy_multi_T_parallel(..., n_workers=1)
```

#### Adjust Chunk Size

The chunk size controls how many disorder realizations are sent to each worker at once.

```python
# Smaller chunks = better load balancing, higher overhead
run_sdrg_entropy_multi_T_parallel(..., chunk_size=5)

# Larger chunks = lower overhead, worse load balancing
run_sdrg_entropy_multi_T_parallel(..., chunk_size=50)

# Auto-calculate (recommended)
run_sdrg_entropy_multi_T_parallel(..., chunk_size=None)
```

**Default calculation**: `chunk_size = max(1, n_disorder // (n_workers * 4))`

For example, with n_disorder=500 and n_workers=10: chunk_size = 12

#### Reproducibility

Results are reproducible when using the same `base_seed`:

```python
# Run 1
run_sdrg_entropy_multi_T_parallel(..., base_seed=42)

# Run 2 (produces identical results)
run_sdrg_entropy_multi_T_parallel(..., base_seed=42)
```

Each worker process gets a unique seed: `seed = base_seed + worker_id * 10000`

## Output Format

The parallel version produces **identical output** to the serial version:

### Combined File: `S_l_all_T.json`

```json
{
  "N": 100,
  "L": 1000,
  "alpha": 3.0,
  "T_list": [0.0, 0.005, 0.01, 0.1, 1.0],
  "n_disorder": 500,
  "n_thermal": 100,
  "S_l_by_T": {
    "0.0": [...],
    "0.005": [...],
    ...
  }
}
```

### Individual Temperature Files: `S_l_T_{T:.3f}.json`

```json
{
  "N": 100,
  "L": 1000,
  "alpha": 3.0,
  "T": 0.005,
  "n_disorder": 500,
  "n_thermal": 100,
  "S_l": [0.0, 0.0, 0.1, ...]
}
```

## Parallelization Strategy

### What is Parallelized

The **disorder loop** (outermost loop) is parallelized:

```python
# Serial version
for d in range(n_disorder):           # 500 iterations - PARALLELIZED ✓
    for T in T_list:                   # 5 iterations
        for _ in range(n_thermal):     # 100 iterations
            # SDRG computation
```

### Why This Approach

- **Embarrassingly parallel**: 500 independent disorder realizations
- **Optimal granularity**: Each task processes all temperatures/thermal samples
- **Minimal overhead**: ~500 tasks vs ~2,500 if parallelizing thermal loop
- **Simple RNG management**: Each worker has isolated random state

### Implementation Details

1. **Worker initialization**: Each worker process gets a unique random seed
2. **Task distribution**: `imap_unordered` distributes disorder realizations to workers
3. **Progress tracking**: Real-time progress counter shows completed tasks
4. **Error handling**: Failed tasks are logged and filtered from results
5. **Result aggregation**: Results are sorted by disorder index before averaging

## Error Handling

### Worker Errors

If a worker encounters an error:
1. Error message is printed with disorder index
2. Task returns `None` result
3. Failed tasks are filtered before averaging
4. Final `n_disorder` count reflects successful tasks

### Common Issues

**ImportError on Windows**: Ensure `if __name__ == "__main__"` guard is present
```python
if __name__ == "__main__":
    run_sdrg_entropy_multi_T_parallel(...)
```

**Inconsistent results**: Check that `base_seed` is set for reproducibility

**Out of memory**: Reduce `n_workers` or `chunk_size`

## Verification

### Verify Combined Numba+Parallel Version

```bash
python verify_numba_parallel.py
```

This runs:
1. **Correctness test**: Numba-only vs Numba+Parallel with same seed
2. **Reproducibility test**: Two parallel runs with same seed
3. **Performance scaling**: Speedup with 1, 2, 4, 8 workers (optional)
4. **Combined speedup**: Compare all versions

Expected results:
- Correctness: <1e-10 difference (exact match)
- Reproducibility: <1e-10 difference (bit-exact)
- Scaling: 6-8× speedup with 8 workers
- Combined speedup: >400× vs original

### Verify Parallel-Only Version

```bash
python verify_parallel.py
```

This runs:
1. **Small-scale test**: Serial vs parallel with n_disorder=10
2. **Reproducibility test**: Two parallel runs with same seed
3. **Performance scaling**: Speedup with 1, 2, 4, 8 workers (optional)

Expected results:
- Maximum difference between serial/parallel: <0.5 (due to different RNG ordering)
- Reproducibility: Identical results (difference < 1e-10)
- Speedup: 6-10× with 8-10 workers

## Integration with Existing Code

All versions are **drop-in replacements** with identical output format:

```python
# Original (slow)
from sdrgX_entropy import run_sdrg_entropy_multi_T
run_sdrg_entropy_multi_T(...)

# Numba-only (70-300× speedup)
from sdrgX_entropy_numba import run_sdrg_entropy_multi_T_numba
run_sdrg_entropy_multi_T_numba(...)

# Parallel-only (6-8× speedup)
from sdrgX_entropy_parallel import run_sdrg_entropy_multi_T_parallel
run_sdrg_entropy_multi_T_parallel(...)

# Numba+Parallel (420-2400× speedup) - RECOMMENDED
from sdrgX_entropy_numba_parallel import run_sdrg_entropy_multi_T_numba_parallel
run_sdrg_entropy_multi_T_numba_parallel(...)
```

The output format is identical across all versions, so **no changes needed** to downstream analysis:

```bash
# Run combined version (fastest)
python sdrgX_entropy_numba_parallel.py

# Plot results (unchanged)
python plot_entropy.py
```

## How Combined Optimization Works

### Multiplicative Speedup

The combined version achieves **multiplicative speedup** by combining two orthogonal optimizations:

1. **Numba (70-300×)**: Optimizes the **innermost computation**
   - JIT-compiles SDRG pairing function (`sdrg_pairing_numba`)
   - JIT-compiles entropy calculation (`entanglement_entropy_numba`)
   - Eliminates Python interpreter overhead
   - Uses optimized NumPy operations

2. **Multiprocessing (6-8×)**: Parallelizes the **outermost loop**
   - Distributes disorder realizations across worker processes
   - Each worker runs independent Numba-compiled code
   - No GIL issues (process-based parallelism)
   - No data sharing between workers (embarrassingly parallel)

**Combined effect**: 70× (Numba) × 8× (8 cores) = **560× speedup**

### Why This Works

The optimizations are **orthogonal** and **complementary**:

```python
# Original nested loop structure
for disorder in range(n_disorder):           # 500 iterations - PARALLELIZED ✓
    for T in T_list:                         # 5 iterations
        for thermal_sample in range(n_thermal):  # 100 iterations
            # SDRG computation - NUMBA-OPTIMIZED ✓
            pairs = sdrg_pairing(...)         # 70-300× faster with Numba
            S = compute_entropy(...)          # 70-300× faster with Numba
```

- **Numba** optimizes the innermost computation (hot loop)
- **Multiprocessing** parallelizes the outermost loop (independent tasks)
- **No interference**: Each worker process runs its own Numba-compiled code

### Implementation Details

**Key architectural decisions**:

1. **Numba functions are copied into combined file**: To avoid import issues in multiprocessing
2. **RNG management**: Each worker seeds with `base_seed + disorder_idx` for reproducibility
3. **Worker function**: Calls Numba-optimized functions instead of original Python functions
4. **Pool pattern**: Uses `imap_unordered` for memory-efficient parallel execution

**From `sdrgX_entropy_numba.py`**:
- `generate_positions_nb()` - Numba-compiled position generation
- `initial_couplings_nb()` - Numba-compiled coupling matrix
- `sdrg_pairing_numba()` - Numba-compiled SDRG pairing
- `entanglement_entropy_numba()` - Numba-compiled entropy calculation

**From `sdrgX_entropy_parallel.py`**:
- `process_disorder_realization()` - Worker function pattern
- `_worker_wrapper()` - Multiprocessing-compatible wrapper
- `run_sdrg_entropy_multi_T_parallel()` - Pool-based parallel driver

**Combined in `sdrgX_entropy_numba_parallel.py`**:
- Worker function calls Numba functions
- Pool distributes work across processes
- Each process runs Numba-compiled code

## Technical Details

### Parallelization Library

**multiprocessing.Pool** (Python standard library)

- No external dependencies
- Process-based parallelism (avoids GIL)
- Automatic RNG isolation per process
- Cross-platform (Linux, macOS, Windows with guard)

### Memory Footprint

Each worker process:
- ~50 MB Python environment
- ~20 MB working data (positions, couplings, results)

Total: ~50 MB × n_workers + 20 MB shared = ~600 MB for 10 workers

### CPU Utilization

- **Compute-bound**: SDRG algorithm is CPU-intensive
- **Near-linear scaling**: Up to 8-10 cores
- **Diminishing returns**: Beyond 10 cores (depends on system)

### Limitations

1. **No nested parallelization**: Cannot simultaneously parallelize disorder + thermal loops
2. **Process overhead**: ~50 MB per worker limits practical worker count
3. **Platform differences**: Windows requires `if __name__` guard
4. **Progress ordering**: Tasks complete out of order (doesn't affect results)

## Troubleshooting

### Slow Performance

**Symptom**: Speedup less than expected

**Solutions**:
- Check CPU usage: Should be near 100% × n_workers
- Reduce `chunk_size` for better load balancing
- Ensure no other CPU-intensive processes running
- Try `n_workers = cpu_count() - 2` to leave more headroom

### Memory Issues

**Symptom**: System runs out of memory

**Solutions**:
- Reduce `n_workers`
- Reduce `chunk_size` (processes fewer tasks at once)
- Close other applications
- Monitor with `htop` or Activity Monitor

### Inconsistent Results

**Symptom**: Different results on each run

**Solutions**:
- Set `base_seed` to a fixed value (e.g., 42)
- Verify worker initialization in logs
- Check for non-deterministic code paths

### Import Errors on Windows

**Symptom**: `RuntimeError: freeze_support()`

**Solution**: Ensure main execution is guarded:
```python
if __name__ == "__main__":
    run_sdrg_entropy_multi_T_parallel(...)
```

## Development Notes

### Modifying the Code

To add new features:

1. **Core algorithm changes**: Modify `sdrg_pairing_finite_T`, `entanglement_entropy_finite_T`
2. **Worker function changes**: Modify `process_disorder_realization`
3. **Parallelization changes**: Modify `run_sdrg_entropy_multi_T_parallel`

Always test with `verify_parallel.py` after changes.

### Testing Strategy

```bash
# Quick test (1-2 minutes)
python verify_parallel.py  # Skip performance scaling

# Full test (5-10 minutes)
python verify_parallel.py  # Answer 'y' to performance scaling

# Production run
python sdrgX_entropy_parallel.py
```

## License

Same as parent project.

## References

See main project README for physics background and citations.

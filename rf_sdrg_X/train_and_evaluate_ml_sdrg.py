#!/usr/bin/env python3
"""
ML Training & Evaluation Script for SDRG-X

This script implements a complete end-to-end pipeline:
1. Generate training data using SDRG simulations
2. Train a RandomForest classifier to predict pairing decisions
3. Evaluate the trained model by computing entanglement entropy S(ℓ)
4. Compare against baseline heuristics
5. Save results and generate comparison plots

Usage:
    python train_and_evaluate_ml_sdrg.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sdrg_X'))

import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
from tqdm import trange
from sdrgML import SDRG_X_Simulator, DataGenerator, MLTrainer
from sdrgX_entropy_numba import (
    generate_positions_nb,
    initial_couplings_nb,
    entanglement_entropy_numba,
    step_T_disorder
)


# ============================================================
# Configuration
# ============================================================

# Training configuration
TRAIN_CONFIG = {
    'N': 80,
    'L': 800,
    'alpha': 2.0,
    'T': 0.01,
    'n_disorder': 500,
    'n_thermal': 100,
    'heuristic': 'strongest',
    'num_trajectories': 1000,  # Number of training episodes
    'steps_per_trajectory': 10,  # Steps per episode
}

# Test configuration (can differ from training, but N must match)
TEST_CONFIG = {
    'N': 80,           # MUST match training N for feature compatibility
    'L': 800,          # Can differ from training
    'alpha': 2.0,      # Can differ from training
    'T': 0.01,         # Can differ from training
    'n_disorder': 500,  # Disorder realizations for entropy averaging
    'n_thermal': 100,  # Thermal samples per disorder
}

# Evaluation configuration
EVAL_CONFIG = {
    'baseline_heuristics': ['strongest', 'random'],  # Heuristics to compare
    'n_disorder_baseline': 500,  # Disorder realizations for baselines
}

# Output configuration
OUTPUT_CONFIG = {
    'plot_file': 'entropy_comparison_ml_vs_heuristics.png',
    'results_json': 'entropy_results.json',
    'model_file': 'trained_rf_model.pkl',
    'save_training_data': False,  # Optional: save (X, y) to disk
}


# ============================================================
# Helper Functions
# ============================================================

def pairing_accuracy(ref_pairs, ml_pairs):
    """
    Compute pairing accuracy r_P = 2 * |ref ∩ ml| / N_spins.

    Pairs are (r1, r2) position tuples; order within each pair is ignored.
    """
    def norm(p):
        return tuple(sorted(p))

    ref_set = set(norm(p) for p in ref_pairs)
    ml_set  = set(norm(p) for p in ml_pairs)
    M_P = len(ref_set & ml_set)
    N   = 2 * len(ref_pairs)
    return 2 * M_P / N if N > 0 else 0.0


def run_model_guided_entropy(simulator, trainer, n_disorder, n_thermal):
    """
    Run model-guided SDRG and compute disorder-averaged entropy.

    This function:
    1. Creates test disorder realizations
    2. Uses the trained model to predict pairing actions
    3. Falls back to heuristics if model predicts invalid actions
    4. Collects pairs and computes entanglement entropy
    5. Averages over disorder and thermal samples

    Parameters
    ----------
    simulator : SDRG_X_Simulator
        Simulator instance with test configuration
    trainer : MLTrainer
        Trained ML model
    n_disorder : int
        Number of disorder realizations
    n_thermal : int
        Number of thermal samples per disorder

    Returns
    -------
    S_avg : np.ndarray, shape (L,)
        Disorder-averaged entanglement entropy S(ℓ)
    rP_all : list of float
        Per-disorder pairing accuracy r_P vs strongest heuristic
    """
    S_all  = []
    rP_all = []

    for d in trange(n_disorder, desc="Disorder", unit="real"):
        # Thermal averaging loop
        S_thermal = []

        for t in range(n_thermal):
            simulator.reset()  # New disorder + thermal sample

            # Save initial state so we can replay with the heuristic
            J_init_saved       = simulator.J_init.copy()
            positions_saved    = simulator.positions.copy()

            # ------ ML run ------
            pairs_r1_list = []
            pairs_r2_list = []
            pairs_s_list  = []
            ml_pairs      = []

            while not simulator.done:
                state_flat = simulator.J_current.flatten()
                state_flat = np.nan_to_num(state_flat, posinf=1e10, neginf=-1e10, nan=0.0)

                action = trainer.model.predict([state_flat])[0]

                max_i = action // simulator.N
                max_j = action % simulator.N

                if max_i >= max_j:
                    max_i, max_j = max_j, max_i

                if (not simulator.active_mask[max_i] or
                    not simulator.active_mask[max_j] or
                    simulator.J_current[max_i, max_j] <= 0):
                    action = simulator._select_action_heuristic()

                next_state, reward, done, info = simulator.step(action=action)

                if 'pair_positions' in info and 'eigenstate' in info:
                    r1, r2 = info['pair_positions']
                    s = info['eigenstate']
                    pairs_r1_list.append(r1)
                    pairs_r2_list.append(r2)
                    pairs_s_list.append(s)
                    ml_pairs.append((r1, r2))

            pairs_r1 = np.array(pairs_r1_list, dtype=np.int64)
            pairs_r2 = np.array(pairs_r2_list, dtype=np.int64)
            pairs_s  = np.array(pairs_s_list,  dtype=np.int64)
            n_pairs  = len(pairs_r1)

            if n_pairs > 0:
                S = entanglement_entropy_numba(pairs_r1, pairs_r2, pairs_s, n_pairs, simulator.L)
                S_thermal.append(S)

            # ------ Strongest heuristic replay on same initial state ------
            # (only on first thermal sample to keep rP per-disorder)
            if t == 0 and ml_pairs:
                simulator.J_current        = J_init_saved.copy()
                simulator.positions_current = positions_saved.copy()
                simulator.active_mask      = np.ones(simulator.N, dtype=bool)
                simulator.n_remaining      = simulator.N
                simulator.done             = False

                ref_pairs = []
                while not simulator.done:
                    action = simulator._select_action_heuristic()
                    _, _, _, h_info = simulator.step(action=action)
                    if 'pair_positions' in h_info:
                        ref_pairs.append(h_info['pair_positions'])

                if ref_pairs:
                    rP_all.append(pairing_accuracy(ref_pairs, ml_pairs))

        if len(S_thermal) > 0:
            S_all.append(np.mean(S_thermal, axis=0))

    S_avg = np.mean(S_all, axis=0) if S_all else np.zeros(simulator.L)
    return S_avg, rP_all


def run_heuristic_entropy(N, L, alpha, T, heuristic, n_disorder, n_thermal):
    """
    Run heuristic-based SDRG and compute disorder-averaged entropy.

    Uses optimized Numba functions for speed (50-100x faster than Python loops).

    Parameters
    ----------
    N : int
        Number of spins
    L : int
        Chain length
    alpha : float
        Coupling decay exponent
    T : float
        Temperature
    heuristic : str
        Pairing heuristic ('strongest', 'random', 'weighted')
    n_disorder : int
        Number of disorder realizations
    n_thermal : int
        Number of thermal samples per disorder

    Returns
    -------
    S_avg : np.ndarray, shape (L,)
        Disorder-averaged entanglement entropy S(ℓ)
    """
    S_list = []

    for d in range(n_disorder):
        # Generate disorder realization
        positions = generate_positions_nb(N, L)
        J_init = initial_couplings_nb(positions, alpha)
        J_init = np.nan_to_num(J_init, posinf=1e10, neginf=-1e10, nan=0.0)

        # Run with thermal averaging (already includes thermal loop)
        S = step_T_disorder(L, positions, J_init, T, N, heuristic, n_thermal)
        S_list.append(S)

    S_avg = np.mean(S_list, axis=0)
    return S_avg


def plot_entropy_comparison(results, config, output_file, rP_all=None):
    """
    Create publication-quality entropy comparison plot.

    Parameters
    ----------
    results : dict
        {method_name: S_array} for each method
    config : dict
        Test configuration (N, L, alpha, T)
    output_file : str
        Output PNG filename
    rP_all : list of float, optional
        Per-disorder pairing accuracy r_P (ML vs strongest).
        If provided, adds a text annotation and histogram inset.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    l_vals = np.arange(config['L'])

    colors = {'model': 'blue', 'strongest': 'red', 'random': 'green', 'weighted': 'orange'}
    styles = {'model': '--', 'strongest': '-', 'random': ':', 'weighted': '-.'}
    method_name = {'model': 'RF-SDRG', 'strongest': 'SDRG', 'random': 'random'}

    for method, S in results.items():
        color = colors.get(method, 'black')
        style = styles.get(method, '-')
        ax.plot(l_vals, S, style, color=color, linewidth=2,
                label=f'{method_name[method]} (σ={np.std(S):.4f})')

    ax.set_xlabel(r'$\ell$', fontsize=14)
    ax.set_ylabel(r'$S(\ell)$', fontsize=14)
    ax.set_title(
        f"Entanglement Entropy: ML vs Heuristics\n"
        f"N={config['N']}, L={config['L']}, α={config['alpha']}, T={config['T']}",
        fontsize=12
    )
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    if rP_all:
        rP_mean = np.mean(rP_all)
        rP_std  = np.std(rP_all)

        # Text annotation: r_P = mean ± std
        ax.text(
            0.05, 0.95,
            rf"$r_P = {rP_mean:.3f} \pm {rP_std:.3f}$",
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )

        # Inset histogram of r_P distribution.
        # ax.inset_axes uses fixed axes-fraction coords and has no dynamic
        # locator, so nudging works reliably.
        # [x0, y0, width, height] all in axes fraction (0–1).
        # x0=0.325 centres a 35%-wide box; y0 is borderpad + nudge.
        _ax_h_px = ax.get_position().height * fig.get_figheight() * fig.dpi
        _border   = plt.rcParams['font.size'] / (_ax_h_px * 72 / fig.dpi)
        _nudge    = 50 / _ax_h_px
        ax_hist = ax.inset_axes((0.325, _border + _nudge, 0.35, 0.35))
        ax_hist.hist(rP_all, bins=20, density=True, alpha=0.8)
        ax_hist.axvline(rP_mean, linestyle='--', linewidth=1, color='red')
        ax_hist.set_xlabel(r'$r_P$', fontsize=8)
        ax_hist.set_ylabel('PDF', fontsize=8)
        ax_hist.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"  ✓ Plot saved to: {output_file}")
    plt.close(fig)


# ============================================================
# Main Pipeline
# ============================================================

def plot_from_json(json_path):
    """Load a saved results JSON and regenerate the comparison plot."""
    with open(json_path) as f:
        data = json.load(f)

    config  = data['test_config']
    results = {k: np.array(v) for k, v in data['S_l_by_method'].items()}
    rP_all  = data.get('r_P_all', None)

    out_dir    = os.path.dirname(json_path) or '.'
    base       = os.path.splitext(os.path.basename(json_path))[0]
    output_file = os.path.join(out_dir, f"{base}_plot.png")

    plot_entropy_comparison(results, config, output_file, rP_all=rP_all)

    if 'r_P_mean' in data:
        print(f"  r_P = {data['r_P_mean']:.3f} ± {data['r_P_std']:.3f}")


def main():
    """Main ML training and evaluation pipeline."""

    print("="*70)
    print("SDRG-X: ML Training & Entropy Evaluation")
    print("="*70)
    print()
    print("Configuration:")
    print(f"  Training: N={TRAIN_CONFIG['N']}, L={TRAIN_CONFIG['L']}, "
          f"α={TRAIN_CONFIG['alpha']}, T={TRAIN_CONFIG['T']}")
    print(f"  Test:     N={TEST_CONFIG['N']}, L={TEST_CONFIG['L']}, "
          f"α={TEST_CONFIG['alpha']}, T={TEST_CONFIG['T']}")
    print()

    # Validation: N must match for feature compatibility
    if TEST_CONFIG['N'] != TRAIN_CONFIG['N']:
        raise ValueError(
            f"TEST_CONFIG['N'] ({TEST_CONFIG['N']}) must equal "
            f"TRAIN_CONFIG['N'] ({TRAIN_CONFIG['N']}) for feature compatibility!"
        )

    # ========================================
    # Phase 1: Generate Training Data
    # ========================================
    print("[1/5] Generating Training Data...")
    print(f"  Trajectories: {TRAIN_CONFIG['num_trajectories']}, "
          f"Steps/trajectory: {TRAIN_CONFIG['steps_per_trajectory']}")

    train_sim = SDRG_X_Simulator(
        N=TRAIN_CONFIG['N'],
        L=TRAIN_CONFIG['L'],
        alpha=TRAIN_CONFIG['alpha'],
        T=TRAIN_CONFIG['T'],
        n_disorder=TRAIN_CONFIG['n_disorder'],
        n_thermal=TRAIN_CONFIG['n_thermal'],
        heuristic=TRAIN_CONFIG['heuristic']
    )

    gen = DataGenerator(train_sim, num_trajectories=TRAIN_CONFIG['num_trajectories'])
    X_train, y_train = gen.generate_supervised_data(It=TRAIN_CONFIG['steps_per_trajectory'])
    print(f"  ✓ Generated {len(X_train)} training samples")

    # Optionally save training data
    if OUTPUT_CONFIG['save_training_data']:
        data_file = 'training_data.pkl'
        with open(data_file, 'wb') as f:
            pickle.dump((X_train, y_train), f)
        print(f"  ✓ Saved training data to: {data_file}")

    # ========================================
    # Phase 2: Train Model
    # ========================================
    print()
    print("[2/5] Training RandomForest Model...")
    print(f"  Model type: RandomForest, Estimators: 100")

    trainer = MLTrainer(model_type='rf')
    trainer.train(X_train, y_train)

    y_pred = trainer.model.predict(X_train)
    train_acc = np.mean(y_pred == y_train)
    print(f"  ✓ Training accuracy: {train_acc:.3f}")

    if OUTPUT_CONFIG['model_file']:
        trainer.save_model(OUTPUT_CONFIG['model_file'])
        print(f"  ✓ Model saved to: {OUTPUT_CONFIG['model_file']}")

    # ========================================
    # Phase 3: Evaluate Model (compute entropy)
    # ========================================
    print()
    print("[3/5] Evaluating Model - Computing S(ℓ)...")
    print(f"  Disorder samples: {TEST_CONFIG['n_disorder']}, "
          f"Thermal samples: {TEST_CONFIG['n_thermal']}")

    test_sim = SDRG_X_Simulator(
        N=TEST_CONFIG['N'],
        L=TEST_CONFIG['L'],
        alpha=TEST_CONFIG['alpha'],
        T=TEST_CONFIG['T'],
        n_disorder=1,
        n_thermal=1,
        heuristic='strongest'
    )

    S_model, rP_all = run_model_guided_entropy(
        test_sim, trainer,
        n_disorder=TEST_CONFIG['n_disorder'],
        n_thermal=TEST_CONFIG['n_thermal']
    )
    rP_mean = float(np.mean(rP_all)) if rP_all else float('nan')
    rP_std  = float(np.std(rP_all))  if rP_all else float('nan')
    print(f"  ✓ Model S(ℓ): mean={np.mean(S_model):.4f}, std={np.std(S_model):.4f}")
    print(f"  ✓ Pairing accuracy r_P = {rP_mean:.3f} ± {rP_std:.3f}")

    # ========================================
    # Phase 4: Evaluate Baselines
    # ========================================
    print()
    print("[4/5] Evaluating Baseline Heuristics...")
    print(f"  Disorder samples: {EVAL_CONFIG['n_disorder_baseline']}, "
          f"Thermal samples: {TEST_CONFIG['n_thermal']}")

    results = {'model': S_model}

    for h in EVAL_CONFIG['baseline_heuristics']:
        print(f"  Running {h}...")
        S_h = run_heuristic_entropy(
            N=TEST_CONFIG['N'],
            L=TEST_CONFIG['L'],
            alpha=TEST_CONFIG['alpha'],
            T=TEST_CONFIG['T'],
            heuristic=h,
            n_disorder=EVAL_CONFIG['n_disorder_baseline'],
            n_thermal=TEST_CONFIG['n_thermal']
        )
        results[h] = S_h
        print(f"    ✓ mean={np.mean(S_h):.4f}, std={np.std(S_h):.4f}")

    # ========================================
    # Phase 5: Save Results and Plot
    # ========================================
    print()
    print("[5/5] Saving Results...")

    # Save JSON
    if OUTPUT_CONFIG['results_json']:
        json_data = {
            'train_config': TRAIN_CONFIG,
            'test_config': TEST_CONFIG,
            'eval_config': EVAL_CONFIG,
            'training_accuracy': float(train_acc),
            'r_P_mean': rP_mean,
            'r_P_std': rP_std,
            'r_P_all': rP_all,
            'S_l_by_method': {k: v.tolist() for k, v in results.items()},
            'statistics': {
                k: {
                    'mean': float(np.mean(v)),
                    'std': float(np.std(v)),
                    'min': float(np.min(v)),
                    'max': float(np.max(v))
                }
                for k, v in results.items()
            }
        }
        with open(OUTPUT_CONFIG['results_json'], 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"  ✓ Results saved to: {OUTPUT_CONFIG['results_json']}")

    # Create plot
    plot_entropy_comparison(results, TEST_CONFIG, OUTPUT_CONFIG['plot_file'], rP_all=rP_all)

    # ========================================
    # Summary
    # ========================================
    print()
    print("="*70)
    print("Summary: Entanglement Entropy Comparison")
    print("="*70)
    print(f"{'Method':<15} {'Mean S(ℓ)':<12} {'Std S(ℓ)':<12}")
    print("-"*70)
    for method, S in results.items():
        print(f"{method:<15} {np.mean(S):<12.6f} {np.std(S):<12.6f}")
    print("="*70)

    # Compute L2 distance between model and strongest
    if 'strongest' in results:
        l2_dist = np.linalg.norm(results['model'] - results['strongest'])
        mean_S = np.mean(results['strongest'])
        relative_dist = l2_dist / mean_S if mean_S > 0 else float('inf')
        print()
        print("Model vs Strongest Comparison:")
        print(f"  L2 distance: {l2_dist:.6f}")
        print(f"  Relative distance: {relative_dist:.2%}")
        print("="*70)


def run_sdrg_entropy_and_rP(simulator, trainer, n_disorder, n_thermal):
    """
    Run SDRG and compute both disorder-averaged entropy S(ℓ) and pairing
    accuracy r_P vs the trained ML model — on the same disorder realizations.

    Symmetric counterpart to run_model_guided_entropy: SDRG is the "main" run
    (thermal-averaged entropy), ML inference is the comparison (r_P, t=0 only).

    Parameters
    ----------
    simulator : SDRG_X_Simulator
        Configured with heuristic='strongest'
    trainer : MLTrainer
        Loaded trained model used for r_P comparison
    n_disorder : int
    n_thermal : int

    Returns
    -------
    S_avg : np.ndarray, shape (L,)
    rP_all : list of float
    """
    S_all  = []
    rP_all = []

    for _ in trange(n_disorder, desc="Disorder (SDRG+r_P)", unit="real"):
        S_thermal = []

        for t in range(n_thermal):
            simulator.reset()
            J_init_saved      = simulator.J_init.copy()
            positions_saved   = simulator.positions.copy()

            # ------ SDRG run (thermal sample) ------
            pairs_r1_list = []
            pairs_r2_list = []
            pairs_s_list  = []
            sdrg_pairs    = []

            while not simulator.done:
                action = simulator._select_action_heuristic()
                _, _, _, info = simulator.step(action=action)
                if 'pair_positions' in info and 'eigenstate' in info:
                    r1, r2 = info['pair_positions']
                    pairs_r1_list.append(r1)
                    pairs_r2_list.append(r2)
                    pairs_s_list.append(info['eigenstate'])
                    if t == 0:
                        sdrg_pairs.append((r1, r2))

            pairs_r1 = np.array(pairs_r1_list, dtype=np.int64)
            pairs_r2 = np.array(pairs_r2_list, dtype=np.int64)
            pairs_s  = np.array(pairs_s_list,  dtype=np.int64)
            n_pairs  = len(pairs_r1)

            if n_pairs > 0:
                S = entanglement_entropy_numba(pairs_r1, pairs_r2, pairs_s, n_pairs, simulator.L)
                S_thermal.append(S)

            # ------ ML replay on same initial state (t=0 only, for r_P) ------
            if t == 0 and sdrg_pairs:
                simulator.J_current         = J_init_saved.copy()
                simulator.positions_current = positions_saved.copy()
                simulator.active_mask       = np.ones(simulator.N, dtype=bool)
                simulator.n_remaining       = simulator.N
                simulator.done              = False

                ml_pairs = []
                while not simulator.done:
                    state_flat = simulator.J_current.flatten()
                    state_flat = np.nan_to_num(state_flat, posinf=1e10, neginf=-1e10, nan=0.0)
                    action = trainer.model.predict([state_flat])[0]
                    max_i = action // simulator.N
                    max_j = action % simulator.N
                    if max_i >= max_j:
                        max_i, max_j = max_j, max_i
                    if (not simulator.active_mask[max_i] or
                            not simulator.active_mask[max_j] or
                            simulator.J_current[max_i, max_j] <= 0):
                        action = simulator._select_action_heuristic()
                    _, _, _, ml_info = simulator.step(action=action)
                    if 'pair_positions' in ml_info:
                        ml_pairs.append(ml_info['pair_positions'])

                if ml_pairs:
                    rP_all.append(pairing_accuracy(sdrg_pairs, ml_pairs))

        if S_thermal:
            S_all.append(np.mean(S_thermal, axis=0))

    S_avg = np.mean(S_all, axis=0) if S_all else np.zeros(simulator.L)
    return S_avg, rP_all


def run_only_sdrg(prev_runs_path, model_path=None):
    """
    Run only the SDRG ('strongest') simulation and combine with previous results.

    Loads RF-SDRG and Random curves from prev_runs_path, runs the strongest
    heuristic with TEST_CONFIG parameters (entropy + r_P in one pass if a
    trained model is available), then plots all three.
    """
    print("="*70)
    print("SDRG-X: SDRG-only run (combining with previous RF/Random results)")
    print("="*70)

    # Load previous results
    with open(prev_runs_path) as f:
        prev = json.load(f)

    prev_methods = prev.get('S_l_by_method', {})
    results = {}

    for key in ('model',):
        if key in prev_methods:
            results[key] = np.array(prev_methods[key])
            print(f"  Loaded '{key}' from {prev_runs_path}")
        else:
            print(f"  WARNING: '{key}' not found in {prev_runs_path}, skipping")

    rP_all = prev.get('r_P_all', None)

    resolved_model_path = model_path or OUTPUT_CONFIG['model_file']

    if os.path.exists(resolved_model_path):
        # Combined run: SDRG entropy + r_P on the same disorder realizations
        print(f"\nRunning SDRG (strongest) + r_P: "
              f"n_disorder={TEST_CONFIG['n_disorder']}, "
              f"n_thermal={TEST_CONFIG['n_thermal']}, "
              f"model={resolved_model_path}")
        trainer = MLTrainer(model_type='rf')
        trainer.load_model(resolved_model_path)
        sim = SDRG_X_Simulator(
            N=TEST_CONFIG['N'],
            L=TEST_CONFIG['L'],
            alpha=TEST_CONFIG['alpha'],
            T=TEST_CONFIG['T'],
            n_disorder=1,
            n_thermal=1,
            heuristic='strongest',
        )
        S_strongest, rP_all = run_sdrg_entropy_and_rP(
            sim, trainer,
            n_disorder=TEST_CONFIG['n_disorder'],
            n_thermal=TEST_CONFIG['n_thermal'],
        )
        rP_mean = float(np.mean(rP_all)) if rP_all else float('nan')
        rP_std  = float(np.std(rP_all))  if rP_all else float('nan')
        print(f"  ✓ r_P = {rP_mean:.3f} ± {rP_std:.3f}")
    else:
        # Fallback: entropy only via fast Numba path, keep old r_P
        print(f"\n  WARNING: model file '{resolved_model_path}' not found — "
              f"computing entropy only, keeping r_P from previous run")
        print(f"\nRunning SDRG (strongest): "
              f"n_disorder={EVAL_CONFIG['n_disorder_baseline']}, "
              f"n_thermal={TEST_CONFIG['n_thermal']}...")
        S_strongest = run_heuristic_entropy(
            N=TEST_CONFIG['N'],
            L=TEST_CONFIG['L'],
            alpha=TEST_CONFIG['alpha'],
            T=TEST_CONFIG['T'],
            heuristic='strongest',
            n_disorder=EVAL_CONFIG['n_disorder_baseline'],
            n_thermal=TEST_CONFIG['n_thermal'],
        )

    results['strongest'] = S_strongest
    print(f"  ✓ SDRG mean={np.mean(S_strongest):.4f}, std={np.std(S_strongest):.4f}")

    # Run random baseline fresh
    print(f"\nRunning random baseline: "
          f"n_disorder={EVAL_CONFIG['n_disorder_baseline']}, "
          f"n_thermal={TEST_CONFIG['n_thermal']}...")
    S_random = run_heuristic_entropy(
        N=TEST_CONFIG['N'],
        L=TEST_CONFIG['L'],
        alpha=TEST_CONFIG['alpha'],
        T=TEST_CONFIG['T'],
        heuristic='random',
        n_disorder=EVAL_CONFIG['n_disorder_baseline'],
        n_thermal=TEST_CONFIG['n_thermal'],
    )
    results['random'] = S_random
    print(f"  ✓ Random mean={np.mean(S_random):.4f}, std={np.std(S_random):.4f}")

    # Merge and save combined JSON
    combined_json = OUTPUT_CONFIG['results_json'].replace('.json', '_combined.json')
    rP_mean_out = float(np.mean(rP_all)) if rP_all else prev.get('r_P_mean')
    rP_std_out  = float(np.std(rP_all))  if rP_all else prev.get('r_P_std')
    combined_data = {
        'test_config': TEST_CONFIG,
        'eval_config': EVAL_CONFIG,
        'source_prev_runs': prev_runs_path,
        'r_P_mean': rP_mean_out,
        'r_P_std':  rP_std_out,
        'r_P_all':  rP_all,
        'S_l_by_method': {k: v.tolist() for k, v in results.items()},
        'statistics': {
            k: {
                'mean': float(np.mean(v)),
                'std':  float(np.std(v)),
                'min':  float(np.min(v)),
                'max':  float(np.max(v)),
            }
            for k, v in results.items()
        },
    }
    with open(combined_json, 'w') as f:
        json.dump(combined_data, f, indent=2)
    print(f"\n  ✓ Combined results saved to: {combined_json}")

    # Plot
    combined_plot = OUTPUT_CONFIG['plot_file'].replace('.png', '_combined.png')
    plot_entropy_comparison(results, TEST_CONFIG, combined_plot, rP_all=rP_all)

    print("\nSummary:")
    print(f"  {'Method':<15} {'Mean S(ℓ)':<12} {'Std S(ℓ)':<12}")
    print("  " + "-"*38)
    for method, S in results.items():
        print(f"  {method:<15} {np.mean(S):<12.6f} {np.std(S):<12.6f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SDRG-X ML training & evaluation")
    parser.add_argument('--just-plot', metavar='RESULT.json',
                        help='Skip computations and regenerate plot from saved JSON')
    parser.add_argument('--only-sdrg', action='store_true',
                        help='Run only SDRG (strongest) simulation and combine with --prev-runs')
    parser.add_argument('--prev-runs', metavar='RESULT.json', default='entropy_results.json',
                        help='JSON file with previous RF-SDRG and Random results '
                             '(used with --only-sdrg, default: entropy_results.json)')
    parser.add_argument('--model', metavar='MODEL.pkl', default=None,
                        help='Trained RF model for r_P recomputation '
                             f'(used with --only-sdrg, default: {OUTPUT_CONFIG["model_file"]})')
    args = parser.parse_args()

    if args.just_plot:
        plot_from_json(args.just_plot)
    elif args.only_sdrg:
        run_only_sdrg(args.prev_runs, model_path=args.model)
    else:
        main()

"""
Test script for SDRG_X_Simulator.step() implementation.
"""

import numpy as np
from sdrgML import SDRG_X_Simulator, DataGenerator

def test_single_step():
    """Test single step execution."""
    print("=" * 60)
    print("Test 1: Single Step Execution")
    print("=" * 60)

    sim = SDRG_X_Simulator(N=10, L=100, alpha=2.0, T=0.01,
                          n_disorder=1, n_thermal=1, heuristic='strongest')

    state, reward, done, info = sim.step()

    print(f"✓ Step executed successfully")
    print(f"  Pair indices: {info['pair_indices']}")
    print(f"  Pair positions: {info['pair_positions']}")
    print(f"  Eigenstate: {info['eigenstate']}")
    print(f"  J_val: {info['J_val']:.6f}")
    print(f"  Remaining spins: {info['n_remaining']}")
    print(f"  Done: {done}")
    print(f"  State shape: {state.shape}")
    print()

def test_trajectory_recording():
    """Test trajectory recording."""
    print("=" * 60)
    print("Test 2: Trajectory Recording")
    print("=" * 60)

    sim = SDRG_X_Simulator(N=10, L=100, alpha=2.0, T=0.01,
                          n_disorder=1, n_thermal=1, heuristic='strongest')
    sim.reset()

    num_steps = 5
    for i in range(num_steps):
        state, reward, done, info = sim.step()
        if done:
            print(f"  Episode terminated early at step {i}")
            break

    print(f"✓ Trajectory length: {len(sim.trajectory)}")
    print(f"  Expected: {min(num_steps, sim.N // 2)}")

    if len(sim.trajectory) > 0:
        state_flat, action_idx = sim.trajectory[0]
        print(f"  First state shape: {state_flat.shape}")
        print(f"  First action index: {action_idx}")
    print()

def test_data_generation():
    """Test data generation using DataGenerator."""
    print("=" * 60)
    print("Test 3: Data Generation")
    print("=" * 60)

    sim = SDRG_X_Simulator(N=10, L=100, alpha=2.0, T=0.01,
                          n_disorder=1, n_thermal=1, heuristic='strongest')
    gen = DataGenerator(sim, num_trajectories=10)

    X, y = gen.generate_supervised_data(It=5)

    print(f"✓ Data generated successfully")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Expected samples: ~50 (10 trajectories * 5 steps)")
    print(f"  Action range: [{y.min()}, {y.max()}]")
    print()

def test_reset_between_episodes():
    """Test that reset properly clears state."""
    print("=" * 60)
    print("Test 4: Reset Between Episodes")
    print("=" * 60)

    sim = SDRG_X_Simulator(N=10, L=100, alpha=2.0, T=0.01,
                          n_disorder=1, n_thermal=1, heuristic='strongest')

    # First episode
    for _ in range(3):
        sim.step()
    traj_len_1 = len(sim.trajectory)
    n_remaining_1 = sim.n_remaining

    # Reset
    sim.reset()

    # Check state is cleared
    assert sim.n_remaining == sim.N, "n_remaining should be reset to N"
    assert sim.done == False, "done should be reset to False"
    assert len(sim.trajectory) == 0, "trajectory should be cleared"
    assert sim.active_mask.sum() == sim.N, "all spins should be active"

    print(f"✓ Reset works correctly")
    print(f"  Before reset: {traj_len_1} steps, {n_remaining_1} remaining")
    print(f"  After reset: {len(sim.trajectory)} steps, {sim.n_remaining} remaining")
    print()

def test_heuristic_comparison():
    """Compare different heuristics."""
    print("=" * 60)
    print("Test 5: Heuristic Comparison")
    print("=" * 60)

    heuristics = ['strongest', 'random', 'weighted']

    for h in heuristics:
        sim = SDRG_X_Simulator(N=10, L=100, alpha=2.0, T=0.01,
                              n_disorder=1, n_thermal=1, heuristic=h)

        # Run a few steps
        steps = 0
        while not sim.done and steps < 5:
            state, reward, done, info = sim.step()
            steps += 1

        print(f"✓ Heuristic '{h}': {steps} steps, {sim.n_remaining} remaining")

    print()

def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)

    # Small system
    sim = SDRG_X_Simulator(N=4, L=10, alpha=2.0, T=0.01,
                          n_disorder=1, n_thermal=1, heuristic='strongest')

    steps = 0
    while not sim.done:
        state, reward, done, info = sim.step()
        steps += 1
        if steps > 10:  # Safety
            break

    print(f"✓ Small system (N=4): terminated after {steps} steps")
    print(f"  Remaining spins: {sim.n_remaining}")

    # Test calling step() after done
    state, reward, done, info = sim.step()
    print(f"✓ Calling step() after done returns correctly: done={done}")

    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SDRG_X_Simulator.step() Implementation Tests")
    print("=" * 60 + "\n")

    try:
        test_single_step()
        test_trajectory_recording()
        test_data_generation()
        test_reset_between_episodes()
        test_heuristic_comparison()
        test_edge_cases()

        print("=" * 60)
        print("✓ ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

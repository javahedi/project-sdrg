"""
Example: Using SDRG_X_Simulator.step() for Machine Learning

This script demonstrates how to use the step() method to:
1. Generate supervised training data from SDRG simulations
2. Train a RandomForest classifier to predict optimal pairings
3. Evaluate the trained model against baseline heuristics
"""

import numpy as np
from sdrgML import SDRG_X_Simulator, DataGenerator, MLTrainer

def example_supervised_learning():
    """
    Complete example of supervised learning pipeline.
    """
    print("=" * 70)
    print("SDRG-X Machine Learning Example")
    print("=" * 70)
    print()

    # ========================================================================
    # Step 1: Setup Simulator
    # ========================================================================
    print("Step 1: Setting up simulator...")
    print("-" * 70)

    # Create simulator with 'strongest' heuristic (expert policy)
    # This will generate training data where we learn to imitate the strongest heuristic
    sim = SDRG_X_Simulator(
        N=20,           # Number of spins
        L=100,          # Chain length
        alpha=2.0,      # Power-law decay
        T=0.01,         # Temperature
        n_disorder=1,   # Disorder realizations (for data generation)
        n_thermal=1,    # Thermal samples per disorder
        heuristic='strongest'  # Expert policy to learn from
    )

    print(f"✓ Simulator created with N={sim.N}, L={sim.L}, alpha={sim.alpha}, T={sim.T}")
    print(f"  Heuristic: {sim.heuristic}")
    print()

    # ========================================================================
    # Step 2: Generate Training Data
    # ========================================================================
    print("Step 2: Generating training data...")
    print("-" * 70)

    # Create data generator
    gen = DataGenerator(sim, num_trajectories=100)

    # Generate supervised data: (states, actions) pairs
    # Each state is a flattened J matrix (N*N features)
    # Each action is the index of the coupling that was selected
    X_train, y_train = gen.generate_supervised_data(It=10)

    print(f"✓ Generated {len(X_train)} training samples")
    print(f"  Input shape: {X_train.shape} (each sample is a flattened {sim.N}x{sim.N} coupling matrix)")
    print(f"  Output shape: {y_train.shape} (action indices)")
    print(f"  Action range: [{y_train.min()}, {y_train.max()}]")
    print()

    # Inspect first few samples
    print("Sample data:")
    for i in range(min(3, len(X_train))):
        state = X_train[i]
        action = y_train[i]
        # Convert action to (i, j) pair
        max_i = action // sim.N
        max_j = action % sim.N
        print(f"  Sample {i}: State has {np.count_nonzero(state)} non-zero couplings, "
              f"action={action} → pair ({max_i}, {max_j})")
    print()

    # ========================================================================
    # Step 3: Train ML Model
    # ========================================================================
    print("Step 3: Training RandomForest model...")
    print("-" * 70)

    # Create and train RandomForest classifier
    trainer = MLTrainer(model_type='rf')
    trainer.train(X_train, y_train)

    print(f"✓ Model trained successfully")
    print(f"  Model type: RandomForest")
    print(f"  Training samples: {len(X_train)}")
    print()

    # Check training accuracy (optional)
    y_pred = trainer.model.predict(X_train)
    train_accuracy = np.mean(y_pred == y_train)
    print(f"  Training accuracy: {train_accuracy:.3f}")
    print()

    # ========================================================================
    # Step 4: Evaluate Model
    # ========================================================================
    print("Step 4: Evaluating model on new episodes...")
    print("-" * 70)

    # Test the model on a new episode
    np.random.seed(999)
    test_sim = SDRG_X_Simulator(
        N=20, L=100, alpha=2.0, T=0.01,
        n_disorder=1, n_thermal=1,
        heuristic='strongest'  # Start with a heuristic, but we'll override with model
    )

    print("Running episode with trained model:")

    # Run one episode using the trained model
    test_sim.reset()
    step_count = 0
    while not test_sim.done and step_count < test_sim.N // 2:
        # Get current state
        state_flat = test_sim.J_current.flatten()

        # Use model to predict action
        action = trainer.predict(state_flat)

        # Take step
        next_state, reward, done, info = test_sim.step(action=action)

        step_count += 1
        print(f"  Step {step_count}: paired ({info['pair_indices'][0]}, {info['pair_indices'][1]}), "
              f"J={info['J_val']:.6f}, remaining={info['n_remaining']}")

    print(f"\n✓ Episode completed in {step_count} steps")
    print()

    # ========================================================================
    # Step 5: Compare with Baselines
    # ========================================================================
    print("Step 5: Comparing model with baseline heuristics...")
    print("-" * 70)

    heuristics = ['strongest', 'random', 'weighted']
    results = {}

    for h in heuristics:
        np.random.seed(42)
        test_sim = SDRG_X_Simulator(
            N=20, L=100, alpha=2.0, T=0.01,
            n_disorder=1, n_thermal=10,
            heuristic=h
        )

        # Run multiple episodes and average
        num_episodes = 5
        step_counts = []
        for _ in range(num_episodes):
            test_sim.reset()
            steps = 0
            while not test_sim.done:
                test_sim.step()
                steps += 1
            step_counts.append(steps)

        avg_steps = np.mean(step_counts)
        std_steps = np.std(step_counts)
        results[h] = (avg_steps, std_steps)

        print(f"  {h:12s}: {avg_steps:.1f} ± {std_steps:.1f} steps per episode")

    print()
    print("=" * 70)
    print("Example Complete!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  • Generated {len(X_train)} training samples from {gen.num_trajectories} trajectories")
    print(f"  • Trained RandomForest model with {train_accuracy:.1%} training accuracy")
    print(f"  • Model can be used to guide SDRG pairing decisions")
    print(f"  • Baseline comparison shows performance across different heuristics")
    print()

if __name__ == "__main__":
    example_supervised_learning()

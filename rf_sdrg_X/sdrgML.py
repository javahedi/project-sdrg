import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'sdrg_X'))

from matplotlib import pyplot as plt
import numpy as np
from utils import initial_couplings
from sdrgX_entropy_numba import initial_couplings_nb, generate_positions_nb, step_T_disorder, sample_pair_state_nb

class SDRG_X_Simulator:
    """
    Simulator for SDRG-X process, designed for ML data generation.
    """
    def __init__(self, N, L, alpha, T, n_disorder, n_thermal, heuristic='strongest'):
        self.N = N          # number of sites per realisation
        self.L = L          # length, number of spins (positions)
        self.alpha = float(alpha)  # alpha
        self.T = T          # temperature
        self.n_disorder = n_disorder  # number of disorder realizations
        self.n_thermal = n_thermal    # number of thermal realizations per disorder
        self.n_samples = n_thermal if T > 0 else 1  # samples per disorder
        self.heuristic = heuristic    # pairing heuristic
        self.trajectory = []
        # Step tracking variables
        self.J_current = None        # Current coupling matrix (mutable)
        self.positions_current = None  # Current positions array
        self.active_mask = None      # Boolean mask of active (unpaired) spins
        self.n_remaining = 0         # Number of unpaired spins
        self.done = False            # Episode termination flag
        self.reset()

    def reset(self):
        # Reset the simulator to initial state
        self.positions = generate_positions_nb(self.N, self.L)
        self.J_init = initial_couplings_nb(self.positions, self.alpha)
        # Sanitize J to prevent numerical issues
        self.J_init = np.nan_to_num(self.J_init, posinf=1e10, neginf=-1e10, nan=0.0)
        # Initialize step tracking state
        self.J_current = self.J_init.copy()  # Working copy
        self.positions_current = self.positions.copy()
        self.active_mask = np.ones(self.N, dtype=bool)  # All spins initially active
        self.n_remaining = self.N
        self.done = False
        self.trajectory = []  # Clear trajectory for new episode

    def run_full(self, n_disorder_run=None):
        # Run a full SDRG-X simulation for It iterations
        # This should implement the full pairing and state sampling logic
        # Storage for all temperatures
        S_by_T = []
        if n_disorder_run is None:
            n_disorder_run = self.n_disorder  # default to one full pass through disorder realizations

        for d in range(n_disorder_run):
            print(f"Disorder realization {d}")

            # Generate random positions using Numba-compiled function
            # (ensures consistent RNG across all operations)
            positions = generate_positions_nb(self.N, self.L)

            # Initialize coupling matrix using Numba-compiled function
            J_init = initial_couplings_nb(positions, self.alpha)

            # Sanitize J to prevent numerical issues
            J_init = np.nan_to_num(J_init, posinf=1e10, neginf=-1e10, nan=0.0)

            S_T = step_T_disorder(self.L, positions, J_init, self.T, self.N, self.heuristic, self.n_thermal)
            S_by_T.append(S_T)

        return np.mean(S_by_T, axis=0)

    def step(self, action=None):
        """
        Perform one SDRG pairing step.

        Parameters
        ----------
        action : int or None
            Flattened index of the coupling to pair. If None, uses self.heuristic.

        Returns
        -------
        next_state : np.ndarray
            Flattened J matrix after the step
        reward : float
            Reward signal (placeholder: 0.0)
        done : bool
            Whether the episode has terminated
        info : dict
            Additional information about the step
        """
        # 1. Check if already done
        if self.done:
            return self.J_current.flatten(), 0.0, True, {}

        # 2. Record state BEFORE action (for ML training)
        state_flat = self.J_current.flatten().copy()

        # 3. Select action (pair to decimate)
        if action is None:
            # Use heuristic to select action
            action_idx = self._select_action_heuristic()
        else:
            action_idx = action

        # 4. Convert flat index to (i, j) indices
        max_i = action_idx // self.N
        max_j = action_idx % self.N

        # Ensure i < j (upper triangular)
        if max_i >= max_j:
            max_i, max_j = max_j, max_i

        # 5. Get coupling value
        J_val = self.J_current[max_i, max_j]

        # 6. Check if valid action
        if J_val <= 0 or not self.active_mask[max_i] or not self.active_mask[max_j]:
            # Invalid action - terminate episode
            self.done = True
            return self.J_current.flatten(), -1.0, True, {'error': 'invalid_action'}

        # 7. Sample eigenstate (thermal average)
        s = sample_pair_state_nb(J_val, self.T)

        # 8. Update trajectory (state, action) for ML training
        self.trajectory.append((state_flat, action_idx))

        # 9. Update J matrix (zero out paired spins)
        self.J_current[max_i, :] = 0.0
        self.J_current[:, max_i] = 0.0
        self.J_current[max_j, :] = 0.0
        self.J_current[:, max_j] = 0.0

        # 10. Update active mask
        self.active_mask[max_i] = False
        self.active_mask[max_j] = False
        self.n_remaining -= 2

        # 11. Check termination
        if self.n_remaining <= 1 or np.max(self.J_current) <= 0:
            self.done = True

        # 12. Compute next state
        next_state = self.J_current.flatten()

        # 13. Return (Gym-like API)
        info = {
            'pair_indices': (max_i, max_j),
            'pair_positions': (self.positions_current[max_i], self.positions_current[max_j]),
            'eigenstate': s,
            'n_remaining': self.n_remaining,
            'J_val': J_val
        }

        return next_state, 0.0, self.done, info

    def _select_action_heuristic(self):
        """
        Select action (coupling index) based on self.heuristic.

        Returns
        -------
        action_idx : int
            Flattened index of the selected coupling
        """
        if self.heuristic == 'strongest':
            # Find maximum coupling (greedy)
            action_idx = np.argmax(self.J_current)

        elif self.heuristic == 'random':
            # Weighted random sampling from upper triangle
            upper = np.triu(self.J_current, k=1)  # Extract upper triangle
            upper_flat = upper.flatten()
            total = np.sum(upper_flat)

            if total <= 0:
                # No valid couplings - return dummy action
                return 0

            # Normalize to probabilities
            probs = upper_flat / total
            cum_probs = np.cumsum(probs)
            r = np.random.random()
            action_idx = np.searchsorted(cum_probs, r)

        elif self.heuristic == 'weighted':
            # Similar to random but with different weighting scheme
            upper = np.triu(self.J_current, k=1)
            upper_flat = upper.flatten()
            total = np.sum(upper_flat)

            if total <= 0:
                return 0

            probs = upper_flat / total
            cum_probs = np.cumsum(probs)
            r = np.random.random()
            action_idx = np.searchsorted(cum_probs, r)

        else:
            raise ValueError(f"Unknown heuristic: {self.heuristic}")

        return action_idx


class DataGenerator:
    """
    Class for generating training data from RG simulations.
    """
    def __init__(self, simulator, num_trajectories=100):
        self.simulator = simulator
        self.num_trajectories = num_trajectories

    def generate_supervised_data(self, It=10):
        """
        Generate supervised data: states and actions using step-by-step simulation.
        """
        X = []  # States (flattened J matrices)
        y = []  # Actions (idx)
        for _ in range(self.num_trajectories):
            self.simulator.reset()
            for _ in range(It):
                state, reward, done, info = self.simulator.step()
                if done:
                    break
            # Collect trajectory
            for state_flat, action in self.simulator.trajectory:
                X.append(state_flat)
                y.append(action)
        return np.array(X), np.array(y)

    def generate_rl_data(self, It=10):
        """
        Generate RL data: list of trajectories (states, actions, rewards).
        """
        raise NotImplementedError("RL data generation not implemented yet")

    def save_data(self, X, y, filename='data.pkl'):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump((X, y), f)


class MLTrainer:
    """
    Class for training ML models on RG data.
    """
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.model = None

    def train(self, X, y):
        if self.model_type == 'rf':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X, y)
        elif self.model_type == 'nn':
            # Placeholder for NN
            pass
        else:
            raise ValueError("Unsupported model type")

    def predict(self, state):
        if self.model:
            return self.model.predict([state])[0]
        return None

    def save_model(self, filename='model.pkl'):
        import pickle
        with open(filename, 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, filename='model.pkl'):
        import pickle
        with open(filename, 'rb') as f:
            self.model = pickle.load(f)


class Evaluator:
    """
    Class for evaluating RG heuristics and ML models.
    """
    def __init__(self, simulator, metrics=['mean_S', 'std_S']):
        self.simulator = simulator
        self.metrics = metrics

    def evaluate_heuristics(self, heuristics, n_disorder_run=100):
        results = {}
        for h in heuristics:
            print(f"Evaluating {h}...")
            sim_temp = SDRG_X_Simulator(self.simulator.N, self.simulator.L, self.simulator.a,
                                        self.simulator.T, self.simulator.n_sample, h)
            S = sim_temp.run_full(n_disorder_run=n_disorder_run)
            results[h] = {'metrics': self.compute_metrics(S), 'S': S}
        return results

    def evaluate_model(self, trainer, It=100, num_episodes=10):
        """
        Evaluate a trained model by running step-by-step simulations.
        """
        print("Evaluating model...")
        S_list = []
        for _ in range(num_episodes):
            self.simulator.reset()
            self.simulator.heuristic = 'model'  # Temporarily set
            trajectory = []
            for _ in range(It):
                state_flat = self.simulator.state.flatten()
                state_flat = np.nan_to_num(state_flat, posinf=1e10, neginf=-1e10)
                action = trainer.predict(state_flat)
                next_state, reward, done, info = self.simulator.step(action)
                trajectory.append(info)
                if done:
                    break
            # Compute S from trajectory (simplified)
            # For now, collect positions and ad
            positions = [t['positions'] for t in trajectory if 'positions' in t]
            ad_list = [t['ad'] for t in trajectory if 'ad' in t]
            # Compute S similar to get_S
            Counter = np.zeros(self.simulator.L)
            for pos1, pos2 in positions:
                for l in range(self.simulator.L):
                    if ad_list[len(Counter) % len(ad_list)] in (3,4):  # Approximate
                        if pos1 < l < pos2:
                            Counter[l] += 1
            S_list.append(Counter)
        S_avg = np.mean(S_list, axis=0)
        results = {'model': {'metrics': self.compute_metrics(S_avg), 'S': S_avg}}
        return results

    def compute_metrics(self, S):
        metrics_dict = {}
        if 'mean_S' in self.metrics:
            metrics_dict['mean_S'] = np.mean(S)
        if 'std_S' in self.metrics:
            metrics_dict['std_S'] = np.std(S)
        return metrics_dict

    def plot_comparison(self, results):
        plt.figure(figsize=(10, 6))
        for h, data in results.items():
            S_std = np.std(data['S'])
            plt.plot(data['S'], label=f'{h}, ({S_std:.3f})')
        plt.xlabel(r'$\ell$')
        plt.ylabel(r'$S(\ell)$')
        plt.title('Comparison of RG Heuristics and Models')
        plt.legend()
        plt.show()


def main():
    # Example Usage of Classes with RandomForest
    sim = SDRG_X_Simulator(n_disorder=100, N=20, L=200, alpha=0.2, T=0.01, n_thermal=10, heuristic='strongest')
    gen = DataGenerator(sim, num_trajectories=200)
    trainer = MLTrainer(model_type='rf')
    evaluator = Evaluator(sim)

    # Generate data using strongest heuristic
    X, y = gen.generate_supervised_data(n_disorder_run=10)
    print(f"Generated {len(X)} data points")

    # Train RandomForest
    trainer.train(X, y)

    # Evaluate heuristics and model
    results_heur = evaluator.evaluate_heuristics(['strongest', 'random', 'weighted'], n_disorder_run=50)
    results_model = evaluator.evaluate_model(trainer, It=10, num_episodes=10)

    # Combine results
    results = {**results_heur, **results_model}

    # Plot comparison
    evaluator.plot_comparison(results)


if __name__ == "__main__":
    main()

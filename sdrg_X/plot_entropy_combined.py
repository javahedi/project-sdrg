#!/usr/bin/env python3
"""
Combined Entropy Plotting Script

Combines three sources of entanglement entropy data into a single comparison plot:
1. Theoretical predictions - CSV files from data_stefan/ (Stefan's analytical results)
2. Numerical simulations - JSON files like S_l_all_T.json (SDRG-X simulation data)
3. ML model results - JSON files like entropy_results.json (trained model evaluations)

Usage examples:
    # Theory + Numerical comparison
    python plot_entropy_combined.py \
        --theory data_stefan/S_l_T_0.01.csv \
        --numerical sdrgX_data_numba/S_l_all_T.json \
        --temperature 0.01 \
        --output theory_vs_numerical.png

    # ML model + baseline comparison
    python plot_entropy_combined.py \
        --ml entropy_results.json \
        --methods model strongest \
        --output ml_vs_strongest.png

    # All three sources combined
    python plot_entropy_combined.py \
        --theory data_stefan/S_l_T_0.01.csv \
        --numerical sdrgX_data_numba/S_l_all_T.json \
        --ml entropy_results.json \
        --temperature 0.01 \
        --methods model \
        --output entropy_all_comparison.png
"""

import argparse
import json
import re
import numpy as np
import matplotlib.pyplot as plt


def load_stefan_entropy(path, has_header=True):
    """
    Load theoretical CSV data.

    Parameters:
    -----------
    path : str
        Path to CSV file with columns: l, S(l)
    has_header : bool
        Whether CSV has a header row to skip

    Returns:
    --------
    l : np.ndarray
        Position values
    S : np.ndarray
        Entropy values
    """
    if has_header:
        data = np.loadtxt(path, delimiter=",", skiprows=1)
    else:
        data = np.loadtxt(path, delimiter=",")

    l = data[:, 0]
    S = data[:, 1]
    return l, S


def load_numerical_json(path, temperature=None):
    """
    Load numerical simulation JSON.

    Parameters:
    -----------
    path : str
        Path to numerical simulation JSON file
    temperature : float or None
        If specified, extract only that specific T.
        If None, return all temperatures.

    Returns:
    --------
    data_dict : dict
        {temperature: (l_vals, S_vals)} for each T
    metadata : dict
        Dictionary with keys: N, L, alpha, n_disorder, n_thermal
    """
    with open(path) as f:
        data = json.load(f)

    L = data["L"]
    l_vals = np.arange(L)

    results = {}
    if temperature is not None:
        T_key = str(float(temperature))
        if T_key in data["S_l_by_T"]:
            S = np.array(data["S_l_by_T"][T_key])
            results[float(temperature)] = (l_vals, S)
        else:
            print(f"Warning: Temperature {temperature} not found in numerical data")
    else:
        # Load all temperatures
        for T_key, S_list in data["S_l_by_T"].items():
            T = float(T_key)
            S = np.array(S_list)
            results[T] = (l_vals, S)

    metadata = {
        'N': data['N'],
        'L': data['L'],
        'alpha': data['alpha'],
        'n_disorder': data.get('n_disorder', 'N/A'),
        'n_thermal': data.get('n_thermal', 'N/A')
    }

    return results, metadata


def load_ml_json(path, methods=None):
    """
    Load ML model evaluation JSON.

    Parameters:
    -----------
    path : str
        Path to entropy_results.json
    methods : list of str, optional
        Methods to extract (e.g., ['model', 'strongest'])
        If None, load all available methods

    Returns:
    --------
    data_dict : dict
        {method_name: (l_vals, S_vals)} for each method
    metadata : dict
        test_config parameters plus training_accuracy
    """
    with open(path) as f:
        data = json.load(f)

    test_config = data['test_config']
    L = test_config['L']
    l_vals = np.arange(L)

    results = {}
    for method, S_list in data['S_l_by_method'].items():
        if methods is None or method in methods:
            S = np.array(S_list)
            results[method] = (l_vals, S)

    metadata = {
        'N': test_config['N'],
        'L': test_config['L'],
        'alpha': test_config['alpha'],
        'T': test_config['T'],
        'training_accuracy': data.get('training_accuracy', None)
    }

    return results, metadata


def plot_combined_entropy(
    theoretical_data=None,
    numerical_data=None,
    ml_data=None,
    temperature=None,
    output_file='entropy_combined.png',
    title=None,
    figsize=(10, 6)
):
    """
    Create combined entropy plot from multiple sources.

    Parameters:
    -----------
    theoretical_data : dict or None
        {temperature: (l_vals, S_vals)} from CSV
    numerical_data : dict or None
        {temperature: (l_vals, S_vals)} from simulation JSON
    ml_data : dict or None
        {method: (l_vals, S_vals)} from ML JSON
    temperature : float or None
        Specific temperature to plot (filters multi-T data)
    output_file : str
        Output filename
    title : str or None
        Custom plot title
    figsize : tuple
        Figure size (width, height)
    """
    plt.figure(figsize=figsize)

    # Color scheme
    colors = {
        'theory': 'black',
        'numerical': 'blue',
        'model': 'red',
        'strongest': 'green',
        'random': 'orange',
        'weighted': 'purple'
    }

    # Line styles
    styles = {
        'theory': '--',
        'numerical': '-',
        'model': '-',
        'strongest': ':',
        'random': '-.',
        'weighted': ':'
    }

    # Plot theoretical data
    if theoretical_data:
        for T, (l_vals, S_vals) in theoretical_data.items():
            if temperature is None or abs(T - temperature) < 1e-6:
                label = f'Theory T={T:g}'
                plt.plot(l_vals, S_vals,
                        linestyle=styles['theory'],
                        color=colors['theory'],
                        linewidth=2,
                        label=label,
                        alpha=0.8)

    # Plot numerical simulation data
    if numerical_data:
        for T, (l_vals, S_vals) in numerical_data.items():
            if temperature is None or abs(T - temperature) < 1e-6:
                std_S = np.std(S_vals)
                label = f'SDRG-X T={T:g} (σ={std_S:.4f})'
                plt.plot(l_vals, S_vals,
                        linestyle=styles['numerical'],
                        color=colors['numerical'],
                        linewidth=2,
                        label=label)

    # Plot ML model data
    if ml_data:
        for method, (l_vals, S_vals) in ml_data.items():
            std_S = np.std(S_vals)
            label = f'ML {method} (σ={std_S:.4f})'
            color = colors.get(method, 'gray')
            style = styles.get(method, '-')
            plt.plot(l_vals, S_vals,
                    linestyle=style,
                    color=color,
                    linewidth=2,
                    label=label)

    # Labels and styling
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$S(\ell)$', fontsize=14)

    if title:
        plt.title(title, fontsize=12)

    plt.legend(fontsize=10, loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Plot saved to: {output_file}")
    plt.close()


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Plot combined entanglement entropy from theory, simulation, and ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Theory + Numerical comparison
  %(prog)s --theory data_stefan/S_l_T_0.01.csv \\
           --numerical sdrgX_data_numba/S_l_all_T.json \\
           --temperature 0.01

  # ML model + baseline comparison
  %(prog)s --ml entropy_results.json --methods model strongest

  # All three sources combined
  %(prog)s --theory data_stefan/S_l_T_0.01.csv \\
           --numerical sdrgX_data_numba/S_l_all_T.json \\
           --ml entropy_results.json \\
           --temperature 0.01 --methods model
        """
    )

    parser.add_argument(
        '--theory',
        type=str,
        help='Path to theoretical CSV file (e.g., data_stefan/S_l_T_0.01.csv)'
    )

    parser.add_argument(
        '--numerical',
        type=str,
        help='Path to numerical simulation JSON (e.g., sdrgX_data/S_l_all_T.json)'
    )

    parser.add_argument(
        '--ml',
        type=str,
        help='Path to ML results JSON (e.g., entropy_results.json)'
    )

    parser.add_argument(
        '--temperature', '-T',
        type=float,
        default=None,
        help='Specific temperature to plot (filters multi-T data)'
    )

    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        default=None,
        help='ML methods to plot (e.g., model strongest)'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default='entropy_combined.png',
        help='Output filename (default: entropy_combined.png)'
    )

    parser.add_argument(
        '--title',
        type=str,
        default=None,
        help='Custom plot title'
    )

    parser.add_argument(
        '--figsize',
        type=float,
        nargs=2,
        default=[10, 6],
        help='Figure size (width height) in inches (default: 10 6)'
    )

    return parser.parse_args()


def main():
    """Main function to orchestrate data loading and plotting."""
    args = parse_arguments()

    print("="*70)
    print("Combined Entropy Plotting Script")
    print("="*70)

    # Check that at least one data source is provided
    if not any([args.theory, args.numerical, args.ml]):
        print("\nError: At least one data source must be provided")
        print("Use --theory, --numerical, or --ml")
        print("Run with --help for usage examples")
        return

    # Load theoretical data
    theoretical_data = None
    if args.theory:
        print(f"\nLoading theoretical data from: {args.theory}")
        try:
            l_vals, S_vals = load_stefan_entropy(args.theory)
            # Extract temperature from filename
            match = re.search(r'T_([0-9.]+)\.csv', args.theory)
            if match:
                T = float(match.group(1))
            else:
                T = 0.0  # Default
            theoretical_data = {T: (l_vals, S_vals)}
            print(f"  ✓ Loaded T={T:g}, {len(S_vals)} points")
        except Exception as e:
            print(f"  ✗ Error loading theoretical data: {e}")

    # Load numerical simulation data
    numerical_data = None
    numerical_metadata = None
    if args.numerical:
        print(f"\nLoading numerical simulation from: {args.numerical}")
        try:
            numerical_data, numerical_metadata = load_numerical_json(
                args.numerical,
                temperature=args.temperature
            )
            print(f"  ✓ Loaded {len(numerical_data)} temperature(s)")
            for T in sorted(numerical_data.keys()):
                print(f"    - T={T:g}")
        except Exception as e:
            print(f"  ✗ Error loading numerical data: {e}")

    # Load ML model data
    ml_data = None
    ml_metadata = None
    if args.ml:
        print(f"\nLoading ML results from: {args.ml}")
        try:
            ml_data, ml_metadata = load_ml_json(args.ml, methods=args.methods)
            print(f"  ✓ Loaded {len(ml_data)} method(s)")
            for method in ml_data.keys():
                print(f"    - {method}")
            if ml_metadata['training_accuracy'] is not None:
                print(f"  Training accuracy: {ml_metadata['training_accuracy']:.3f}")
        except Exception as e:
            print(f"  ✗ Error loading ML data: {e}")

    # Check that we successfully loaded at least one data source
    if not any([theoretical_data, numerical_data, ml_data]):
        print("\nError: No data successfully loaded. Exiting.")
        return

    # Generate title if not provided
    title = args.title
    if title is None and (numerical_metadata or ml_metadata):
        meta = ml_metadata or numerical_metadata
        title = (
            f"Entanglement Entropy Comparison\n"
            f"N={meta['N']}, L={meta['L']}, α={meta['alpha']}"
        )
        if 'T' in meta:
            title += f", T={meta['T']}"

    # Create plot
    print(f"\nGenerating plot...")
    try:
        plot_combined_entropy(
            theoretical_data=theoretical_data,
            numerical_data=numerical_data,
            ml_data=ml_data,
            temperature=args.temperature,
            output_file=args.output,
            title=title,
            figsize=tuple(args.figsize)
        )
    except Exception as e:
        print(f"  ✗ Error generating plot: {e}")
        import traceback
        traceback.print_exc()
        return

    print("\n" + "="*70)
    print("Done!")
    print("="*70)


if __name__ == "__main__":
    main()

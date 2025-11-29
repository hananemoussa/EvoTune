#!/usr/bin/env python3
"""
Visualize LLM-SRBench Dataset

This script provides comprehensive visualization of the LLM-SRBench dataset,
including metadata from parquet files and data from the HDF5 file.

Usage:
    # Show all problems in a category
    python visualize_dataset.py --category bio_pop_growth

    # Show details for a specific problem
    python visualize_dataset.py --problem BPG10

    # Plot data for a specific problem
    python visualize_dataset.py --problem BPG10 --plot

    # Compare multiple problems
    python visualize_dataset.py --compare BPG0 BPG10 CRK5

    # Save plots to file
    python visualize_dataset.py --problem BPG10 --plot --save

    # Show summary statistics
    python visualize_dataset.py --summary
"""

import argparse
import json
import os
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import h5py

# Try to import matplotlib (optional for non-plotting usage)
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def get_data_paths():
    """Get paths to the data files."""
    base_paths = [
        Path("llm-srbench-dataset/llm-srbench"),
        Path("../llm-srbench-dataset/llm-srbench"),
        Path("/fs/ess/PAA0201/hananemoussa/EvoTune/llm-srbench-dataset/llm-srbench"),
    ]

    for base in base_paths:
        hdf5_path = base / "lsr_bench_data.hdf5"
        parquet_dir = base / "data"
        if hdf5_path.exists():
            return hdf5_path, parquet_dir

    raise FileNotFoundError("Could not find llm-srbench dataset")


def load_metadata():
    """Load metadata from parquet files."""
    _, parquet_dir = get_data_paths()

    metadata = {}
    parquet_files = {
        'bio_pop_growth': 'lsr_synth_bio_pop_growth-00000-of-00001.parquet',
        'chem_react': 'lsr_synth_chem_react-00000-of-00001.parquet',
        'matsci': 'lsr_synth_matsci-00000-of-00001.parquet',
        'phys_osc': 'lsr_synth_phys_osc-00000-of-00001.parquet',
        'lsr_transform': 'lsr_transform-00000-of-00001.parquet',
    }

    for category, filename in parquet_files.items():
        filepath = parquet_dir / filename
        if filepath.exists():
            df = pd.read_parquet(filepath)
            metadata[category] = []
            for _, row in df.iterrows():
                metadata[category].append({
                    'name': row['name'],
                    'expression': row['expression'],
                    'symbols': list(row['symbols']),
                    'symbol_descs': list(row['symbol_descs']),
                    'symbol_properties': list(row['symbol_properties']),
                })

    return metadata


def load_problem_data(category: str, problem_name: str) -> Dict:
    """Load data for a specific problem from HDF5."""
    hdf5_path, _ = get_data_paths()

    # Determine HDF5 path
    if category == 'lsr_transform':
        base_path = f'/lsr_transform/{problem_name}'
    else:
        base_path = f'/lsr_synth/{category}/{problem_name}'

    data = {}
    with h5py.File(hdf5_path, 'r') as f:
        if base_path not in f:
            raise ValueError(f"Problem {problem_name} not found in category {category}")

        grp = f[base_path]
        for split in grp.keys():
            arr = grp[split][:]
            data[split] = {
                'y': arr[:, 0],
                'X': arr[:, 1:],
                'raw': arr,
            }

    return data


def find_problem_category(problem_name: str, metadata: Dict) -> Optional[str]:
    """Find which category a problem belongs to."""
    for category, problems in metadata.items():
        for p in problems:
            if p['name'] == problem_name:
                return category
    return None


def get_problem_metadata(problem_name: str, metadata: Dict) -> Optional[Dict]:
    """Get metadata for a specific problem."""
    for category, problems in metadata.items():
        for p in problems:
            if p['name'] == problem_name:
                return p
    return None


def print_problem_details(problem_name: str, metadata: Dict, include_data_stats: bool = True):
    """Print detailed information about a problem."""
    category = find_problem_category(problem_name, metadata)
    if category is None:
        print(f"Error: Problem '{problem_name}' not found")
        return

    prob_meta = get_problem_metadata(problem_name, metadata)

    print("=" * 80)
    print(f"Problem: {problem_name}")
    print(f"Category: {category}")
    print("=" * 80)

    # Variables
    print("\n📋 VARIABLES:")
    print("-" * 40)
    symbols = prob_meta['symbols']
    descs = prob_meta['symbol_descs']
    props = prob_meta['symbol_properties']

    prop_map = {'O': 'Output', 'V': 'Variable', 'C': 'Constant'}

    print(f"  Output (y): {symbols[0]}")
    print(f"    Description: {descs[0]}")
    print()
    print(f"  Inputs (X):")
    for i, (sym, desc, prop) in enumerate(zip(symbols[1:], descs[1:], props[1:])):
        print(f"    X[:, {i}] = {sym} ({prop_map.get(prop, prop)})")
        print(f"             {desc}")

    # Ground truth equation
    print("\n📐 GROUND TRUTH EQUATION:")
    print("-" * 40)
    expr = prob_meta['expression']
    # Pretty print long equations
    if len(expr) > 70:
        # Try to break at operators
        formatted = expr
        for op in [' + ', ' - ', ' * ']:
            formatted = formatted.replace(op, f'\n      {op.strip()} ')
        print(f"  {symbols[0]} = ")
        for line in formatted.split('\n'):
            print(f"      {line.strip()}")
    else:
        print(f"  {symbols[0]} = {expr}")

    # Data statistics
    if include_data_stats:
        print("\n📊 DATA STATISTICS:")
        print("-" * 40)
        try:
            data = load_problem_data(category, problem_name)

            for split, split_data in data.items():
                y = split_data['y']
                X = split_data['X']

                print(f"\n  {split.upper()}:")
                print(f"    Samples: {len(y)}")
                print(f"    Features: {X.shape[1]}")
                print()
                print(f"    y ({symbols[0]}):")
                print(f"      min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")

                for i in range(X.shape[1]):
                    sym = symbols[i + 1] if i + 1 < len(symbols) else f"x{i}"
                    print(f"    X[:, {i}] ({sym}):")
                    print(f"      min={X[:, i].min():.4f}, max={X[:, i].max():.4f}, mean={X[:, i].mean():.4f}, std={X[:, i].std():.4f}")

        except Exception as e:
            print(f"  Error loading data: {e}")

    print("\n" + "=" * 80)


def plot_problem(problem_name: str, metadata: Dict, save: bool = False, save_dir: str = "plots"):
    """Create visualizations for a problem."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    category = find_problem_category(problem_name, metadata)
    if category is None:
        print(f"Error: Problem '{problem_name}' not found")
        return

    prob_meta = get_problem_metadata(problem_name, metadata)
    data = load_problem_data(category, problem_name)

    symbols = prob_meta['symbols']
    n_inputs = len(symbols) - 1

    # Create figure
    n_splits = len(data)
    fig = plt.figure(figsize=(6 * n_splits, 4 * (n_inputs + 1)))
    gs = gridspec.GridSpec(n_inputs + 1, n_splits, figure=fig)

    fig.suptitle(f"Problem: {problem_name} ({category})\n{prob_meta['symbol_descs'][0]}",
                 fontsize=14, fontweight='bold')

    for col, (split, split_data) in enumerate(data.items()):
        y = split_data['y']
        X = split_data['X']

        # Plot y distribution
        ax = fig.add_subplot(gs[0, col])
        ax.hist(y, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(f'{symbols[0]} (y)')
        ax.set_ylabel('Count')
        ax.set_title(f'{split.upper()} - {symbols[0]} distribution\n(n={len(y)})')
        ax.axvline(y.mean(), color='red', linestyle='--', label=f'mean={y.mean():.2f}')
        ax.legend()

        # Plot each input vs output
        for i in range(n_inputs):
            ax = fig.add_subplot(gs[i + 1, col])

            # Subsample if too many points
            if len(y) > 2000:
                idx = np.random.choice(len(y), 2000, replace=False)
                x_plot, y_plot = X[idx, i], y[idx]
            else:
                x_plot, y_plot = X[:, i], y

            ax.scatter(x_plot, y_plot, alpha=0.3, s=10, c='steelblue')
            ax.set_xlabel(f'{symbols[i + 1]} (X[:, {i}])')
            ax.set_ylabel(f'{symbols[0]} (y)')
            ax.set_title(f'{split.upper()} - {symbols[0]} vs {symbols[i + 1]}')

    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"{problem_name}_visualization.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    else:
        plt.show()

    plt.close()


def plot_comparison(problem_names: List[str], metadata: Dict, save: bool = False, save_dir: str = "plots"):
    """Compare multiple problems side by side."""
    if not HAS_MATPLOTLIB:
        print("Error: matplotlib is required for plotting. Install with: pip install matplotlib")
        return

    n_problems = len(problem_names)
    fig, axes = plt.subplots(2, n_problems, figsize=(5 * n_problems, 8))

    if n_problems == 1:
        axes = axes.reshape(-1, 1)

    fig.suptitle("Problem Comparison - Training Data", fontsize=14, fontweight='bold')

    for col, problem_name in enumerate(problem_names):
        category = find_problem_category(problem_name, metadata)
        if category is None:
            print(f"Warning: Problem '{problem_name}' not found, skipping")
            continue

        prob_meta = get_problem_metadata(problem_name, metadata)
        data = load_problem_data(category, problem_name)

        symbols = prob_meta['symbols']
        train_data = data.get('train', list(data.values())[0])
        y = train_data['y']
        X = train_data['X']

        # y distribution
        ax = axes[0, col]
        ax.hist(y, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(f'{symbols[0]}')
        ax.set_ylabel('Count')
        ax.set_title(f'{problem_name}\n{category}')

        # X[:, 0] vs y scatter
        ax = axes[1, col]
        if len(y) > 1000:
            idx = np.random.choice(len(y), 1000, replace=False)
            ax.scatter(X[idx, 0], y[idx], alpha=0.3, s=10, c='steelblue')
        else:
            ax.scatter(X[:, 0], y, alpha=0.3, s=10, c='steelblue')
        ax.set_xlabel(f'{symbols[1]}')
        ax.set_ylabel(f'{symbols[0]}')

    plt.tight_layout()

    if save:
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, f"comparison_{'_'.join(problem_names)}.png")
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"Saved plot to: {filepath}")
    else:
        plt.show()

    plt.close()


def print_category_summary(category: str, metadata: Dict):
    """Print summary of all problems in a category."""
    if category not in metadata:
        print(f"Error: Category '{category}' not found")
        print(f"Available categories: {list(metadata.keys())}")
        return

    problems = metadata[category]

    print("=" * 100)
    print(f"Category: {category} ({len(problems)} problems)")
    print("=" * 100)

    # Header
    print(f"\n{'Name':<15} {'Inputs':<8} {'Output':<30} {'Equation Preview':<50}")
    print("-" * 100)

    for p in sorted(problems, key=lambda x: x['name']):
        n_inputs = len(p['symbols']) - 1
        output = p['symbol_descs'][0][:28]
        expr_preview = p['expression'][:48] + "..." if len(p['expression']) > 48 else p['expression']
        print(f"{p['name']:<15} {n_inputs:<8} {output:<30} {expr_preview:<50}")

    print()


def print_full_summary(metadata: Dict):
    """Print summary of the entire dataset."""
    print("=" * 80)
    print("LLM-SRBench Dataset Summary")
    print("=" * 80)

    total = 0
    for category, problems in metadata.items():
        total += len(problems)

        # Get unique output descriptions
        outputs = set(p['symbol_descs'][0] for p in problems)
        output_sample = list(outputs)[0][:50]

        # Input count range
        input_counts = [len(p['symbols']) - 1 for p in problems]

        print(f"\n{category}:")
        print(f"  Problems: {len(problems)}")
        print(f"  Inputs: {min(input_counts)}-{max(input_counts)} variables")
        print(f"  Output: {output_sample}...")
        print(f"  Examples: {', '.join(p['name'] for p in problems[:5])}...")

    print(f"\n{'=' * 80}")
    print(f"Total problems: {total}")
    print("=" * 80)

    # HDF5 file info
    try:
        hdf5_path, _ = get_data_paths()
        with h5py.File(hdf5_path, 'r') as f:
            print(f"\nHDF5 file: {hdf5_path}")
            print(f"Top-level groups: {list(f.keys())}")
    except Exception as e:
        print(f"\nError reading HDF5: {e}")


def print_hdf5_structure():
    """Print the full HDF5 file structure."""
    hdf5_path, _ = get_data_paths()

    print("=" * 80)
    print(f"HDF5 File Structure: {hdf5_path}")
    print("=" * 80)

    with h5py.File(hdf5_path, 'r') as f:
        def print_item(name, obj):
            indent = "  " * name.count('/')
            if isinstance(obj, h5py.Dataset):
                print(f"{indent}  {name.split('/')[-1]}: shape={obj.shape}, dtype={obj.dtype}")
            else:
                print(f"{indent}  {name.split('/')[-1]}/")

        f.visititems(print_item)


def print_raw_data(problem_name: str, metadata: Dict, num_rows: int = 10, split: str = None):
    """Print raw data content for a problem."""
    category = find_problem_category(problem_name, metadata)
    if category is None:
        print(f"Error: Problem '{problem_name}' not found")
        return

    prob_meta = get_problem_metadata(problem_name, metadata)
    data = load_problem_data(category, problem_name)
    symbols = prob_meta['symbols']

    print("=" * 80)
    print(f"Raw Data: {problem_name} ({category})")
    print("=" * 80)

    # Print equation
    print(f"\nEquation: {symbols[0]} = {prob_meta['expression']}")
    print()

    # Column headers
    col_names = [f"{symbols[0]} (y)"] + [f"{symbols[i+1]} (X[:,{i}])" for i in range(len(symbols)-1)]

    splits_to_show = [split] if split else list(data.keys())

    for s in splits_to_show:
        if s not in data:
            print(f"Warning: Split '{s}' not found. Available: {list(data.keys())}")
            continue

        split_data = data[s]
        raw = split_data['raw']

        print(f"\n{s.upper()} ({raw.shape[0]} rows x {raw.shape[1]} columns)")
        print("-" * 80)

        # Print header
        header = "  Row  |"
        for i, name in enumerate(col_names):
            header += f" {name[:15]:>15} |"
        print(header)
        print("-" * 80)

        # Print rows
        rows_to_print = min(num_rows, len(raw))
        for i in range(rows_to_print):
            row_str = f" {i:4d}  |"
            for j in range(raw.shape[1]):
                val = raw[i, j]
                if abs(val) < 0.001 or abs(val) >= 10000:
                    row_str += f" {val:15.4e} |"
                else:
                    row_str += f" {val:15.6f} |"
            print(row_str)

        if rows_to_print < len(raw):
            print(f"  ...   ({len(raw) - rows_to_print} more rows)")

        print()


def print_parquet_content(category: str = None):
    """Print content of parquet metadata files."""
    _, parquet_dir = get_data_paths()

    parquet_files = {
        'bio_pop_growth': 'lsr_synth_bio_pop_growth-00000-of-00001.parquet',
        'chem_react': 'lsr_synth_chem_react-00000-of-00001.parquet',
        'matsci': 'lsr_synth_matsci-00000-of-00001.parquet',
        'phys_osc': 'lsr_synth_phys_osc-00000-of-00001.parquet',
        'lsr_transform': 'lsr_transform-00000-of-00001.parquet',
    }

    if category and category not in parquet_files:
        print(f"Error: Category '{category}' not found")
        print(f"Available: {list(parquet_files.keys())}")
        return

    categories_to_show = [category] if category else list(parquet_files.keys())

    for cat in categories_to_show:
        filepath = parquet_dir / parquet_files[cat]
        if not filepath.exists():
            print(f"Warning: File not found: {filepath}")
            continue

        print("=" * 100)
        print(f"Parquet File: {parquet_files[cat]}")
        print("=" * 100)

        df = pd.read_parquet(filepath)
        print(f"\nColumns: {list(df.columns)}")
        print(f"Rows: {len(df)}")
        print()

        for idx, row in df.iterrows():
            print(f"--- {row['name']} ---")
            print(f"  expression: {row['expression']}")
            print(f"  symbols: {list(row['symbols'])}")
            print(f"  symbol_descs: {list(row['symbol_descs'])}")
            print(f"  symbol_properties: {list(row['symbol_properties'])}")
            print()

        print()


def main():
    parser = argparse.ArgumentParser(
        description="View LLM-SRBench dataset content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--problem', '-p', type=str,
                        help='Show details for a specific problem')
    parser.add_argument('--category', '-c', type=str,
                        help='List all problems in a category')
    parser.add_argument('--data', '-d', action='store_true',
                        help='Print raw data values (use with --problem)')
    parser.add_argument('--rows', '-r', type=int, default=10,
                        help='Number of data rows to print (default: 10, use -1 for all)')
    parser.add_argument('--split', '-s', type=str,
                        help='Only show specific split (train/test/ood_test)')
    parser.add_argument('--parquet', action='store_true',
                        help='Print parquet metadata content')
    parser.add_argument('--plot', action='store_true',
                        help='Create visualization plots')
    parser.add_argument('--compare', nargs='+', type=str,
                        help='Compare multiple problems')
    parser.add_argument('--save', action='store_true',
                        help='Save plots to files instead of displaying')
    parser.add_argument('--save-dir', type=str, default='plots',
                        help='Directory to save plots (default: plots)')
    parser.add_argument('--summary', action='store_true',
                        help='Show dataset summary')
    parser.add_argument('--hdf5-structure', action='store_true',
                        help='Show HDF5 file structure')
    parser.add_argument('--no-stats', action='store_true',
                        help='Skip data statistics when showing problem details')

    args = parser.parse_args()

    # Load metadata
    try:
        metadata = load_metadata()
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return 1

    # Handle commands
    if args.hdf5_structure:
        print_hdf5_structure()
    elif args.parquet:
        print_parquet_content(args.category)
    elif args.summary:
        print_full_summary(metadata)
    elif args.category and not args.problem:
        print_category_summary(args.category, metadata)
    elif args.compare:
        if args.plot:
            plot_comparison(args.compare, metadata, save=args.save, save_dir=args.save_dir)
        else:
            for prob in args.compare:
                print_problem_details(prob, metadata, include_data_stats=not args.no_stats)
    elif args.problem:
        if args.data:
            # Print raw data content
            num_rows = args.rows if args.rows > 0 else 999999
            print_raw_data(args.problem, metadata, num_rows=num_rows, split=args.split)
        else:
            print_problem_details(args.problem, metadata, include_data_stats=not args.no_stats)
            if args.plot:
                plot_problem(args.problem, metadata, save=args.save, save_dir=args.save_dir)
    else:
        # Default: show summary
        print_full_summary(metadata)
        print("\nUse --help for more options")
        print("Examples:")
        print("  --problem BPG10           Show problem details and stats")
        print("  --problem BPG10 --data    Print raw data values")
        print("  --problem BPG10 -d -r 50  Print 50 rows of data")
        print("  --problem BPG10 --plot    Show details and visualize")
        print("  --category bio_pop_growth List all problems in category")
        print("  --parquet                 Print all parquet metadata")
        print("  --hdf5-structure          Show HDF5 file structure")

    return 0


if __name__ == "__main__":
    exit(main())

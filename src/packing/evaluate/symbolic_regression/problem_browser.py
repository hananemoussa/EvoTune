#!/usr/bin/env python3
"""
Problem Browser for LLM-SRBench Symbolic Regression Tasks

This utility helps you explore and select problems from the LLM-SRBench benchmark.

Usage:
    # List all categories
    python problem_browser.py --list-categories

    # List all problems in a category
    python problem_browser.py --category bio_pop_growth

    # Show details for a specific problem
    python problem_browser.py --problem BPG0

    # Show problems sorted by complexity (number of terms in equation)
    python problem_browser.py --category bio_pop_growth --sort-by complexity

    # Show problems with specific number of inputs
    python problem_browser.py --num-inputs 2

    # Interactive mode
    python problem_browser.py --interactive
"""

import json
import argparse
import os
from pathlib import Path
from typing import Optional, Dict, List
import re


def get_metadata_path() -> Path:
    """Get path to the problem metadata JSON file."""
    possible_paths = [
        Path("llm-srbench-dataset/problem_metadata.json"),
        Path("../llm-srbench-dataset/problem_metadata.json"),
        Path("/fs/ess/PAA0201/hananemoussa/EvoTune/llm-srbench-dataset/problem_metadata.json"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    raise FileNotFoundError("problem_metadata.json not found. Run the metadata extraction first.")


def load_metadata() -> Dict:
    """Load the problem metadata."""
    path = get_metadata_path()
    with open(path, 'r') as f:
        return json.load(f)


def estimate_complexity(expression: str) -> int:
    """Estimate equation complexity by counting operations and terms."""
    # Count various mathematical operations
    ops = ['+', '-', '*', '/', '**', 'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'Abs']
    count = 0
    for op in ops:
        count += expression.count(op)
    return count


def get_num_inputs(problem: Dict) -> int:
    """Get number of input variables (excluding output)."""
    return len(problem['symbols']) - 1


def format_problem_short(problem: Dict, category: str) -> str:
    """Format problem for short listing."""
    name = problem['name']
    output = problem['symbol_descs'][0]
    inputs = problem['symbol_descs'][1:]
    num_inputs = len(inputs)
    complexity = estimate_complexity(problem['expression'])

    return f"{name:12} | {num_inputs} inputs | complexity: {complexity:3} | {output[:40]}"


def format_problem_detailed(problem: Dict, category: str) -> str:
    """Format problem with full details."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Problem: {problem['name']} (Category: {category})")
    lines.append("=" * 80)

    # Variables
    lines.append("\nVariables:")
    lines.append(f"  Output: {problem['symbols'][0]} = {problem['symbol_descs'][0]}")
    lines.append("  Inputs:")
    for sym, desc, prop in zip(problem['symbols'][1:], problem['symbol_descs'][1:], problem['symbol_properties'][1:]):
        prop_desc = {'V': 'Variable', 'C': 'Constant', 'O': 'Output'}.get(prop, prop)
        lines.append(f"    - {sym} ({prop_desc}): {desc}")

    # Ground truth equation
    lines.append(f"\nGround Truth Equation:")
    expr = problem['expression']
    # Break long expressions
    if len(expr) > 70:
        lines.append(f"  {expr[:70]}")
        for i in range(70, len(expr), 70):
            lines.append(f"  {expr[i:i+70]}")
    else:
        lines.append(f"  {expr}")

    # Complexity estimate
    complexity = estimate_complexity(expr)
    lines.append(f"\nComplexity Score: {complexity}")

    # Command to run
    lines.append(f"\nTo run this problem:")
    lines.append(f"  python src/experiments/main.py task=sr \\")
    lines.append(f"      task.dataset_category={category} \\")
    lines.append(f"      task.problem_name={problem['name']} \\")
    lines.append(f"      model=llama32 train=none prefix=sr_{problem['name'].lower()}")

    lines.append("=" * 80)
    return "\n".join(lines)


def list_categories(metadata: Dict) -> None:
    """List all available categories with problem counts."""
    print("\n" + "=" * 60)
    print("Available Categories")
    print("=" * 60)

    for category, problems in metadata.items():
        print(f"\n{category}:")
        print(f"  Problems: {len(problems)}")

        # Get unique output descriptions
        outputs = set(p['symbol_descs'][0] for p in problems)
        print(f"  Output variable: {list(outputs)[0][:50]}")

        # Get input count range
        input_counts = [get_num_inputs(p) for p in problems]
        print(f"  Input variables: {min(input_counts)}-{max(input_counts)}")

        # Complexity range
        complexities = [estimate_complexity(p['expression']) for p in problems]
        print(f"  Complexity range: {min(complexities)}-{max(complexities)}")


def list_problems_in_category(metadata: Dict, category: str, sort_by: str = 'name') -> None:
    """List all problems in a category."""
    if category not in metadata:
        print(f"Error: Category '{category}' not found.")
        print(f"Available categories: {list(metadata.keys())}")
        return

    problems = metadata[category]

    # Sort
    if sort_by == 'complexity':
        problems = sorted(problems, key=lambda p: estimate_complexity(p['expression']))
    elif sort_by == 'inputs':
        problems = sorted(problems, key=lambda p: get_num_inputs(p))
    else:  # name
        problems = sorted(problems, key=lambda p: p['name'])

    print(f"\n{'=' * 80}")
    print(f"Category: {category} ({len(problems)} problems)")
    print(f"{'=' * 80}")
    print(f"{'Name':12} | Inputs | Complexity | Output Description")
    print("-" * 80)

    for p in problems:
        print(format_problem_short(p, category))


def show_problem(metadata: Dict, problem_name: str) -> None:
    """Show detailed information for a specific problem."""
    # Find the problem
    for category, problems in metadata.items():
        for p in problems:
            if p['name'] == problem_name:
                print(format_problem_detailed(p, category))
                return

    print(f"Error: Problem '{problem_name}' not found.")
    print("Use --list-categories to see available categories, then --category <name> to list problems.")


def filter_by_inputs(metadata: Dict, num_inputs: int) -> None:
    """Show all problems with a specific number of inputs."""
    print(f"\n{'=' * 80}")
    print(f"Problems with {num_inputs} input variable(s)")
    print(f"{'=' * 80}")

    found = False
    for category, problems in metadata.items():
        matching = [p for p in problems if get_num_inputs(p) == num_inputs]
        if matching:
            found = True
            print(f"\n{category} ({len(matching)} problems):")
            print("-" * 60)
            for p in sorted(matching, key=lambda x: estimate_complexity(x['expression'])):
                print(f"  {format_problem_short(p, category)}")

    if not found:
        print(f"No problems found with {num_inputs} inputs.")


def show_simplest_problems(metadata: Dict, top_n: int = 10) -> None:
    """Show the simplest problems across all categories."""
    all_problems = []
    for category, problems in metadata.items():
        for p in problems:
            all_problems.append((category, p, estimate_complexity(p['expression'])))

    # Sort by complexity
    all_problems.sort(key=lambda x: x[2])

    print(f"\n{'=' * 80}")
    print(f"Top {top_n} Simplest Problems (by complexity score)")
    print(f"{'=' * 80}")
    print(f"{'Rank':4} | {'Name':12} | {'Category':15} | {'Cmplx':5} | Inputs | Expression Preview")
    print("-" * 100)

    for i, (category, p, complexity) in enumerate(all_problems[:top_n], 1):
        expr_preview = p['expression'][:35] + "..." if len(p['expression']) > 35 else p['expression']
        num_inputs = get_num_inputs(p)
        print(f"{i:4} | {p['name']:12} | {category:15} | {complexity:5} | {num_inputs:6} | {expr_preview}")


def interactive_mode(metadata: Dict) -> None:
    """Interactive problem browser."""
    print("\n" + "=" * 60)
    print("LLM-SRBench Problem Browser - Interactive Mode")
    print("=" * 60)
    print("\nCommands:")
    print("  categories     - List all categories")
    print("  cat <name>     - List problems in a category")
    print("  show <name>    - Show details for a problem")
    print("  simple [n]     - Show n simplest problems (default: 10)")
    print("  inputs <n>     - Show problems with n inputs")
    print("  help           - Show this help")
    print("  quit           - Exit")
    print()

    while True:
        try:
            cmd = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not cmd:
            continue

        parts = cmd.split()
        command = parts[0].lower()
        args = parts[1:]

        if command in ('quit', 'exit', 'q'):
            print("Exiting.")
            break
        elif command in ('categories', 'cats', 'list'):
            list_categories(metadata)
        elif command in ('cat', 'category'):
            if args:
                list_problems_in_category(metadata, args[0])
            else:
                print("Usage: cat <category_name>")
        elif command in ('show', 'info', 'details'):
            if args:
                show_problem(metadata, args[0])
            else:
                print("Usage: show <problem_name>")
        elif command in ('simple', 'simplest'):
            n = int(args[0]) if args else 10
            show_simplest_problems(metadata, n)
        elif command in ('inputs', 'in'):
            if args:
                filter_by_inputs(metadata, int(args[0]))
            else:
                print("Usage: inputs <number>")
        elif command in ('help', 'h', '?'):
            print("\nCommands:")
            print("  categories     - List all categories")
            print("  cat <name>     - List problems in a category")
            print("  show <name>    - Show details for a problem")
            print("  simple [n]     - Show n simplest problems")
            print("  inputs <n>     - Show problems with n inputs")
            print("  quit           - Exit")
        else:
            print(f"Unknown command: {command}. Type 'help' for available commands.")


def main():
    parser = argparse.ArgumentParser(
        description="Browse and select LLM-SRBench symbolic regression problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list-categories
  %(prog)s --category bio_pop_growth
  %(prog)s --category bio_pop_growth --sort-by complexity
  %(prog)s --problem BPG0
  %(prog)s --num-inputs 2
  %(prog)s --simplest 10
  %(prog)s --interactive
        """
    )

    parser.add_argument('--list-categories', '-l', action='store_true',
                        help='List all available categories')
    parser.add_argument('--category', '-c', type=str,
                        help='List problems in a specific category')
    parser.add_argument('--problem', '-p', type=str,
                        help='Show details for a specific problem')
    parser.add_argument('--sort-by', choices=['name', 'complexity', 'inputs'], default='name',
                        help='Sort problems by (default: name)')
    parser.add_argument('--num-inputs', '-n', type=int,
                        help='Filter problems by number of input variables')
    parser.add_argument('--simplest', '-s', type=int, nargs='?', const=10,
                        help='Show N simplest problems (default: 10)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Start interactive browser')

    args = parser.parse_args()

    # Load metadata
    try:
        metadata = load_metadata()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1

    # Handle commands
    if args.interactive:
        interactive_mode(metadata)
    elif args.list_categories:
        list_categories(metadata)
    elif args.category:
        list_problems_in_category(metadata, args.category, args.sort_by)
    elif args.problem:
        show_problem(metadata, args.problem)
    elif args.num_inputs is not None:
        filter_by_inputs(metadata, args.num_inputs)
    elif args.simplest:
        show_simplest_problems(metadata, args.simplest)
    else:
        # Default: show summary
        print("\nLLM-SRBench Problem Browser")
        print("=" * 40)
        total = sum(len(p) for p in metadata.values())
        print(f"Total problems: {total}")
        print(f"Categories: {len(metadata)}")
        print("\nUse --help for available commands")
        print("Use --interactive for interactive mode")
        print("Use --simplest to see easiest problems to start with")

    return 0


if __name__ == "__main__":
    exit(main())

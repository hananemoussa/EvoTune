"""
Symbolic Regression Task for EvoTune

This module implements the symbolic regression task using datasets from llm-srbench.
The task is to discover mathematical equations that fit the given data points.

Datasets available:
- bio_pop_growth: 24 problems (biology population growth)
- chem_react: 36 problems (chemical reaction kinetics)
- matsci: 25 problems (material science)
- phys_osc: 44 problems (physics oscillators)
- lsr_transform: 111 problems (transformed Feynman equations)

Evaluation metric: Negative NMSE (Normalized Mean Squared Error)
- Score = -NMSE * 100 (higher is better, 0 is perfect)

Data split mapping:
- train: llm-srbench train set (used for evolutionary search scoring)
- trainperturbedset: llm-srbench test set (in-distribution, for periodic evaluation)
- testset: llm-srbench test set (in-distribution, for final evaluation)
- ood_test: llm-srbench ood_test set (out-of-distribution, only for lsr_synth problems)
"""

import numpy as np
import h5py
from pathlib import Path
from omegaconf import DictConfig
import traceback
import os
import json
import logging

from packing.evaluate.registry import TASK_REGISTRY


# ============================================
# DATA LOADING UTILITIES
# ============================================

def get_data_path():
    """Get the path to the HDF5 data file."""
    # Try multiple possible locations
    possible_paths = [
        Path("llm-srbench-dataset/llm-srbench/lsr_bench_data.hdf5"),
        Path("../llm-srbench-dataset/llm-srbench/lsr_bench_data.hdf5"),
        Path("/fs/ess/PAA0201/hananemoussa/EvoTune/llm-srbench-dataset/llm-srbench/lsr_bench_data.hdf5"),
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Default path - will raise error if not found
    return possible_paths[0]


def load_sr_data(dataset_category: str, problem_name: str, split: str):
    """
    Load symbolic regression data from the HDF5 file.

    Args:
        dataset_category: e.g., 'bio_pop_growth', 'chem_react', 'phys_osc', 'matsci', 'lsr_transform'
        problem_name: e.g., 'BPG0', 'CRK0', 'PO0', 'MatSci0', 'I.10.7_1_0'
        split: 'train', 'test', or 'ood_test'

    Returns:
        dict with 'X' (features) and 'y' (target)
    """
    data_path = get_data_path()

    # Map dataset category to HDF5 path
    if dataset_category == 'lsr_transform':
        h5_path = f'/lsr_transform/{problem_name}/{split}'
    else:
        h5_path = f'/lsr_synth/{dataset_category}/{problem_name}/{split}'

    with h5py.File(data_path, 'r') as f:
        if h5_path not in f:
            raise ValueError(f"Path {h5_path} not found in HDF5 file. "
                           f"Available groups: {list(f.keys())}")
        data = f[h5_path][:].astype(np.float64)

    # Column 0 is target (y), remaining columns are features (X)
    return {
        'y': data[:, 0],
        'X': data[:, 1:]
    }


def has_ood_test(dataset_category: str, problem_name: str) -> bool:
    """
    Check if OOD test set exists for the given problem.

    OOD test sets are only available for lsr_synth problems (bio_pop_growth,
    chem_react, phys_osc, matsci), not for lsr_transform problems.
    """
    if dataset_category == 'lsr_transform':
        return False

    data_path = get_data_path()
    h5_path = f'/lsr_synth/{dataset_category}/{problem_name}/ood_test'

    with h5py.File(data_path, 'r') as f:
        return h5_path in f


# ============================================
# 1. DEFINE INITIAL HEURISTIC
# ============================================

def get_initial_func(cfg):
    """
    Returns the initial/baseline function to seed the evolutionary search.
    For symbolic regression, we start with a simple linear combination.
    """
    def equation(X: np.ndarray) -> np.ndarray:
        """Predicts the target variable y given input features X.

        Args:
            X: Input features array of shape (n_samples, n_features)

        Returns:
            y_pred: Predicted values array of shape (n_samples,)
        """
        # Simple baseline: weighted sum of inputs
        # This is a reasonable starting point for many regression problems
        n_features = X.shape[1]
        weights = np.ones(n_features) / n_features
        return np.dot(X, weights)

    return equation, "equation"


# ============================================
# 2. GENERATE INPUT DATA
# ============================================

def generate_input(cfg, set: str):
    """
    Generate or load the dataset for evaluation.

    Args:
        cfg: Hydra config containing:
            - dataset_category: 'bio_pop_growth', 'chem_react', 'phys_osc', 'matsci', or 'lsr_transform'
            - problem_name: e.g., 'BPG0', 'CRK0', 'PO0', 'MatSci0', 'I.10.7_1_0'
        set: One of "train", "trainperturbedset", "testset"

    Returns:
        Dictionary with 'X' and 'y' arrays, plus metadata

    Mapping to llm-srbench splits:
        - "train" -> llm-srbench 'train' (used for evolutionary scoring)
        - "trainperturbedset" -> llm-srbench 'test' (in-distribution, for periodic eval)
        - "testset" -> llm-srbench 'test' (in-distribution, for final eval)
    """
    category = cfg.dataset_category
    problem = cfg.problem_name

    if set == "train":
        data = load_sr_data(category, problem, 'train')
        data['split'] = 'train'
        data['dataset_category'] = category
        data['problem_name'] = problem
        return data

    elif set == "trainperturbedset":
        # Map to in-distribution test set for periodic evaluation
        data = load_sr_data(category, problem, 'test')
        data['split'] = 'test'
        data['dataset_category'] = category
        data['problem_name'] = problem
        return data

    elif set == "testset":
        # In-distribution test set for final evaluation
        data = load_sr_data(category, problem, 'test')
        data['split'] = 'test'
        data['dataset_category'] = category
        data['problem_name'] = problem
        return data

    else:
        raise ValueError(f"Invalid dataset set: {set}")


# ============================================
# 3. EVALUATE FUNCTION (with NMSE)
# ============================================

GENERAL_IMPORTS = '''
import numpy as np
import numpy
import math
from math import pi, e, sqrt, log, log10, exp, sin, cos, tan, asin, acos, atan, sinh, cosh, tanh
import scipy
import scipy.special
import scipy.stats
'''


def compute_nmse(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute Normalized Mean Squared Error.

    NMSE = MSE / Var(y_true)

    Lower is better. NMSE=0 means perfect prediction.
    NMSE=1 means predictions are as good as predicting the mean.
    """
    # Handle NaN/Inf values in predictions
    valid_mask = np.isfinite(y_pred)
    if np.sum(valid_mask) == 0:
        return float('inf')

    y_pred_valid = y_pred[valid_mask]
    y_true_valid = y_true[valid_mask]

    # Compute MSE
    mse = np.mean((y_true_valid - y_pred_valid) ** 2)

    # Compute variance of true values
    var = np.var(y_true_valid)

    if var == 0:
        # If variance is 0, all true values are the same
        return float('inf') if mse > 1e-10 else 0.0

    return float(mse / var)


def evaluate_func(cfg, dataset, function_class):
    """
    Evaluate a generated function on the symbolic regression task.

    Args:
        cfg: Hydra config
        dataset: Output from generate_input() - dict with 'X' and 'y'
        function_class: FunctionClass containing the generated function

    Returns:
        FunctionClass with updated score and eval fields
    """
    equation_func_str = function_class.function_str
    imports = function_class.imports_str

    # Step 1: Parse and compile the function
    try:
        # Create execution environment
        globals_dict = {}
        exec(GENERAL_IMPORTS, globals_dict)
        exec(imports, globals_dict)

        local_dict = {}
        exec(equation_func_str, globals_dict, local_dict)

        func_from_llm = local_dict.get(cfg.function_str_to_extract)

        if func_from_llm is None:
            raise ValueError(f"Function '{cfg.function_str_to_extract}' not found in generated code")

    except Exception as e:
        tb_str = traceback.format_exc()
        function_class.fail_flag = 1
        function_class.score = cfg.task.failed_score
        function_class.true_score = cfg.task.failed_score
        function_class.fail_exception = tb_str
        return function_class

    # Step 2: Evaluate the function on the dataset
    try:
        X = dataset['X']
        y_true = dataset['y']

        # Call the generated function
        y_pred = func_from_llm(X)

        # Ensure output is numpy array with correct shape
        y_pred = np.asarray(y_pred, dtype=np.float64)

        # Handle different output shapes
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        # Check output length matches input
        if len(y_pred) != len(y_true):
            raise ValueError(f"Output length {len(y_pred)} doesn't match expected {len(y_true)}")

        # Compute NMSE
        nmse = compute_nmse(y_pred, y_true)

        # Convert to score (higher is better in EvoTune)
        # Score = -NMSE * 100
        if nmse == float('inf') or nmse > 1e6:
            score = cfg.task.failed_score
        else:
            score = -nmse * 100
            score = round(score, 4)

        function_class.score = score
        function_class.true_score = score
        function_class.fail_flag = 0
        function_class.correct_flag = 1

    except Exception as e:
        tb_str = traceback.format_exc()
        function_class.fail_flag = 1
        function_class.score = cfg.task.failed_score
        function_class.true_score = cfg.task.failed_score
        function_class.fail_exception = tb_str

    return function_class


# ============================================
# 4. PROMPTS AND REGISTRATION
# ============================================

system_prompt = """You are an expert mathematician and scientist specializing in discovering mathematical equations from data.
Your goal is to find a mathematical function that accurately predicts the output variable y from the input features X.

The function signature is:
def equation(X: np.ndarray) -> np.ndarray:
    '''
    Predicts y values from input features X.

    Args:
        X: Input features array of shape (n_samples, n_features)
           Access individual features as X[:, 0], X[:, 1], etc.

    Returns:
        y_pred: Predicted values array of shape (n_samples,)
    '''
    # Your equation implementation here
    return y_pred

Available operations:
- Basic: +, -, *, /, ** (power)
- NumPy: np.sin, np.cos, np.tan, np.exp, np.log, np.sqrt, np.abs, np.tanh, np.sinh, np.cosh
- NumPy: np.power, np.arcsin, np.arccos, np.arctan, np.log10, np.log2
- Array operations: np.sum, np.prod, np.mean along axes

The score is based on Normalized Mean Squared Error (NMSE):
- Score = -NMSE * 100 (higher/less negative is better)
- Score of 0 means perfect prediction
- More negative scores indicate worse predictions

You are an expert in writing Python functions."""

append_prompt = """Based on the functions shown above, create a new equation() function that achieves a better (higher/less negative) score.

Guidelines:
1. Analyze what makes the higher-scoring function better
2. Consider combining different mathematical operations (polynomials, trigonometric, exponential, logarithmic)
3. Look for patterns in how features interact (products, ratios, sums)
4. Try novel mathematical transformations

IMPORTANT:
- Your solution MUST be a single, self-contained function named equation()
- Only the equation() function will be extracted and executed
- Do NOT define helper functions outside equation()
- All logic must be contained within the equation() function itself
- Make sure to return an array of the same length as the input X"""

TASK_REGISTRY.register(
    "sr",  # Task name used in command line
    generate_input=generate_input,
    evaluate_func=evaluate_func,
    get_initial_func=get_initial_func,
    system_prompt=system_prompt,
    append_prompt=append_prompt,
)


# ============================================
# 5. FINAL EVALUATION UTILITIES
# ============================================

def evaluate_function_on_split(func_from_llm, dataset_category: str, problem_name: str, split: str) -> dict:
    """
    Evaluate a compiled function on a specific data split.

    Args:
        func_from_llm: Compiled function to evaluate
        dataset_category: e.g., 'bio_pop_growth', 'chem_react', etc.
        problem_name: e.g., 'BPG0', 'CRK0', etc.
        split: 'train', 'test', or 'ood_test'

    Returns:
        dict with 'nmse', 'score', 'success', and optional 'error'
    """
    try:
        data = load_sr_data(dataset_category, problem_name, split)
        X = data['X']
        y_true = data['y']

        y_pred = func_from_llm(X)
        y_pred = np.asarray(y_pred, dtype=np.float64)

        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()

        if len(y_pred) != len(y_true):
            return {
                'nmse': float('inf'),
                'score': float('-inf'),
                'success': False,
                'error': f"Output length {len(y_pred)} doesn't match expected {len(y_true)}"
            }

        nmse = compute_nmse(y_pred, y_true)
        score = -nmse * 100 if nmse != float('inf') else float('-inf')

        return {
            'nmse': nmse,
            'score': round(score, 4) if score != float('-inf') else score,
            'success': True,
            'n_samples': len(y_true)
        }

    except Exception as e:
        return {
            'nmse': float('inf'),
            'score': float('-inf'),
            'success': False,
            'error': str(e)
        }


def evaluate_best_program_all_splits(cfg, function_str: str, imports_str: str = "") -> dict:
    """
    Evaluate the best program on train, test, and OOD test sets.

    This function should be called at the end of a run to get final metrics.

    Args:
        cfg: Hydra config with dataset_category and problem_name
        function_str: The function code string
        imports_str: Import statements needed by the function

    Returns:
        dict with metrics for each split:
        {
            'train': {'nmse': ..., 'score': ..., 'success': ...},
            'test': {'nmse': ..., 'score': ..., 'success': ...},
            'ood_test': {'nmse': ..., 'score': ..., 'success': ...} or None
        }
    """
    category = cfg.dataset_category if hasattr(cfg, 'dataset_category') else cfg.task.dataset_category
    problem = cfg.problem_name if hasattr(cfg, 'problem_name') else cfg.task.problem_name
    func_to_extract = cfg.function_str_to_extract if hasattr(cfg, 'function_str_to_extract') else cfg.task.function_str_to_extract

    # Compile the function
    try:
        globals_dict = {}
        exec(GENERAL_IMPORTS, globals_dict)
        if imports_str:
            exec(imports_str, globals_dict)

        local_dict = {}
        exec(function_str, globals_dict, local_dict)

        func_from_llm = local_dict.get(func_to_extract)
        if func_from_llm is None:
            return {
                'train': {'success': False, 'error': f"Function '{func_to_extract}' not found"},
                'test': {'success': False, 'error': f"Function '{func_to_extract}' not found"},
                'ood_test': None,
                'compilation_error': True
            }
    except Exception as e:
        return {
            'train': {'success': False, 'error': str(e)},
            'test': {'success': False, 'error': str(e)},
            'ood_test': None,
            'compilation_error': True
        }

    results = {
        'dataset_category': category,
        'problem_name': problem,
        'compilation_error': False
    }

    # Evaluate on train
    results['train'] = evaluate_function_on_split(func_from_llm, category, problem, 'train')

    # Evaluate on test (in-distribution)
    results['test'] = evaluate_function_on_split(func_from_llm, category, problem, 'test')

    # Evaluate on OOD test if available
    if has_ood_test(category, problem):
        results['ood_test'] = evaluate_function_on_split(func_from_llm, category, problem, 'ood_test')
    else:
        results['ood_test'] = None

    return results


def save_final_sr_metrics(cfg, best_function_str: str, imports_str: str, logs_dir: str, round_num: int = -1):
    """
    Evaluate the best program and save final metrics to a JSON file.

    This should be called at the end of a run.

    Args:
        cfg: Hydra config
        best_function_str: The best function code string
        imports_str: Import statements
        logs_dir: Directory to save the metrics file
        round_num: The round number when this was called (-1 for final)

    Returns:
        dict with all metrics
    """
    results = evaluate_best_program_all_splits(cfg, best_function_str, imports_str)
    results['round_num'] = round_num
    results['function_str'] = best_function_str
    results['imports_str'] = imports_str

    # Create metrics directory if needed
    metrics_dir = os.path.join(logs_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)

    # Save to JSON file
    metrics_file = os.path.join(metrics_dir, 'final_sr_metrics.json')

    # Convert any numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    serializable_results = convert_to_serializable(results)

    with open(metrics_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logging.info(f"Final SR metrics saved to {metrics_file}")

    # Log summary
    logging.info("=" * 50)
    logging.info("FINAL SYMBOLIC REGRESSION METRICS")
    logging.info("=" * 50)
    logging.info(f"Problem: {results['dataset_category']}/{results['problem_name']}")

    if results['train']['success']:
        logging.info(f"Train NMSE: {results['train']['nmse']:.6f} (score: {results['train']['score']:.4f})")
    else:
        logging.info(f"Train: FAILED - {results['train'].get('error', 'Unknown error')}")

    if results['test']['success']:
        logging.info(f"Test NMSE:  {results['test']['nmse']:.6f} (score: {results['test']['score']:.4f})")
    else:
        logging.info(f"Test: FAILED - {results['test'].get('error', 'Unknown error')}")

    if results['ood_test'] is not None:
        if results['ood_test']['success']:
            logging.info(f"OOD NMSE:   {results['ood_test']['nmse']:.6f} (score: {results['ood_test']['score']:.4f})")
        else:
            logging.info(f"OOD: FAILED - {results['ood_test'].get('error', 'Unknown error')}")
    else:
        logging.info("OOD Test: Not available for this problem")

    logging.info("=" * 50)

    return results


# ============================================
# TESTING (when run directly)
# ============================================

if __name__ == "__main__":
    from omegaconf import OmegaConf
    from packing.logging.function_class import FunctionClass
    from packing.utils.functions import function_to_string

    # Create test config
    cfg = OmegaConf.create({
        "dataset_category": "bio_pop_growth",
        "problem_name": "BPG0",
        "function_str_to_extract": "equation",
        "task": {
            "failed_score": -10000
        }
    })

    print("=== Testing Symbolic Regression Task ===")

    # Test data loading
    print("\n1. Testing data loading...")
    try:
        train_data = generate_input(cfg, "train")
        print(f"   Train data loaded: X shape = {train_data['X'].shape}, y shape = {train_data['y'].shape}")

        test_data = generate_input(cfg, "testset")
        print(f"   Test data loaded: X shape = {test_data['X'].shape}, y shape = {test_data['y'].shape}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        print("   Make sure the HDF5 file path is correct")

    # Test initial function
    print("\n2. Testing initial function...")
    initial_func, func_name = get_initial_func(cfg)
    print(f"   Function name: {func_name}")

    # Test evaluation
    print("\n3. Testing evaluation...")

    def equation(X: np.ndarray) -> np.ndarray:
        """Simple linear combination baseline."""
        return np.sum(X, axis=1)

    function_str = function_to_string(equation)
    imports_str = "import numpy as np"

    function_class = FunctionClass(function_str, imports_str)

    try:
        result = evaluate_func(cfg, train_data, function_class)
        print(f"   Score: {result.score}")
        print(f"   Fail flag: {result.fail_flag}")
        if result.fail_flag:
            print(f"   Exception: {result.fail_exception}")
    except Exception as e:
        print(f"   Error during evaluation: {e}")

    print("\n=== Test Complete ===")

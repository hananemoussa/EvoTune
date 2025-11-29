# Adding Symbolic Regression Task from LLM-SRBench to EvoTune

This document summarizes the changes made to add symbolic regression tasks from the [LLM-SRBench](https://github.com/deep-symbolic-mathematics/llm-srbench) benchmark to EvoTune.

## Summary

The symbolic regression task enables EvoTune to discover mathematical equations from scientific data using the LLM-SRBench benchmark. The benchmark contains 239 problems across 5 scientific domains:

| Dataset Category | Domain | # Problems | Problem Names |
|-----------------|--------|------------|---------------|
| `bio_pop_growth` | Biology (Population Growth) | 24 | BPG0 - BPG23 |
| `chem_react` | Chemistry (Reaction Kinetics) | 36 | CRK0 - CRK35 |
| `phys_osc` | Physics (Oscillators) | 44 | PO0 - PO43 |
| `matsci` | Material Science | 25 | MatSci0 - MatSci28 |
| `lsr_transform` | Physics (Transformed Feynman) | 111 | I.10.7_1_0, etc. |

## Files Changed/Created

### 1. New Files Created

#### `src/packing/evaluate/symbolic_regression/task_sr.py`
Main task implementation containing:
- `get_initial_func(cfg)` - Returns baseline linear equation function
- `generate_input(cfg, set)` - Loads data from HDF5 file
- `evaluate_func(cfg, dataset, function_class)` - Evaluates equations using NMSE metric
- `load_sr_data(category, problem_name, split)` - Helper to load from HDF5
- `has_ood_test(category, problem_name)` - Check if OOD test set exists
- `compute_nmse(y_pred, y_true)` - Computes Normalized Mean Squared Error
- `evaluate_best_program_all_splits(cfg, function_str, imports_str)` - Evaluate on all splits
- `save_final_sr_metrics(cfg, function_str, imports_str, logs_dir, round_num)` - Save final metrics
- Task registration with `TASK_REGISTRY.register("sr", ...)`

#### `configs/task/sr.yaml`
Task configuration file with:
```yaml
task_name: "sr"
function_str_to_extract: "equation"
dataset_category: "bio_pop_growth"  # Configurable
problem_name: "BPG0"                 # Configurable
failed_score: -10000
timeout_period: 30
mem_limit_gb: 10
```

### 2. Files Modified

#### `src/packing/funsearch/programs_database.py`
Added `get_best_program` property to retrieve the best program across all islands.

#### `src/experiments/main.py`
Added final evaluation at the end of runs for the `sr` task that:
- Retrieves the best program from the program database
- Evaluates it on train, test, and OOD test sets
- Saves metrics to `metrics/final_sr_metrics.json`
- Logs metrics to wandb (if enabled)

### 3. Dependencies Added

The following package was installed in the `evotune` conda environment:
```bash
pip install h5py
```

### 4. Data Location

The LLM-SRBench dataset is located at:
```
llm-srbench-dataset/llm-srbench/lsr_bench_data.hdf5
```

This HDF5 file contains all 239 problems with train/test/ood_test splits.

## Implementation Details

### Data Split Mapping

The mapping between EvoTune's dataset names and LLM-SRBench splits:

| EvoTune Set | LLM-SRBench Split | Purpose |
|-------------|-------------------|---------|
| `train` | `train` | Evolutionary search scoring |
| `trainperturbedset` | `test` | Periodic evaluation (in-distribution) |
| `testset` | `test` | Final evaluation (in-distribution) |
| N/A | `ood_test` | Final evaluation (out-of-distribution) |

**Key design decisions:**
- The evolutionary search scores functions **only on the training set**
- Periodic evaluation during training uses the **in-distribution test set** (not perturbed noise)
- At the end of a run, the best program is evaluated on **train, test, and OOD test** sets
- OOD test is only available for `lsr_synth` problems (not `lsr_transform`)

### Evaluation Metric: NMSE (Normalized Mean Squared Error)

```python
NMSE = MSE(y_pred, y_true) / Var(y_true)
```

- NMSE = 0: Perfect prediction
- NMSE = 1: Predictions as good as predicting the mean
- NMSE > 1: Worse than predicting the mean

The score in EvoTune is: `score = -NMSE * 100` (higher is better)

### Function Signature

The LLM generates functions with this signature:
```python
def equation(X: np.ndarray) -> np.ndarray:
    """
    Args:
        X: Input features array of shape (n_samples, n_features)
           Access features as X[:, 0], X[:, 1], etc.

    Returns:
        y_pred: Predicted values array of shape (n_samples,)
    """
    # Mathematical equation implementation
    return y_pred
```

### Final Metrics Output

At the end of a run, the following file is created:
```
{logs_dir}/metrics/final_sr_metrics.json
```

Example content:
```json
{
  "dataset_category": "bio_pop_growth",
  "problem_name": "BPG0",
  "compilation_error": false,
  "train": {
    "nmse": 0.0523,
    "score": -5.23,
    "success": true,
    "n_samples": 4000
  },
  "test": {
    "nmse": 0.0612,
    "score": -6.12,
    "success": true,
    "n_samples": 500
  },
  "ood_test": {
    "nmse": 0.1842,
    "score": -18.42,
    "success": true,
    "n_samples": 500
  },
  "round_num": 1000,
  "function_str": "def equation(X):\n    ...",
  "imports_str": "import numpy as np"
}
```

For `lsr_transform` problems, `ood_test` will be `null` since they don't have OOD test sets.

## Usage

### Running with Default Problem (BPG0)

```bash
# FunSearch baseline (no fine-tuning)
python src/experiments/main.py \
    task=sr \
    model=llama32 \
    train=none \
    prefix=funsearch_sr_bpg0 \
    num_rounds=100

# EvoTune with DPO
python src/experiments/main.py \
    task=sr \
    model=llama32 \
    train=dpo \
    prefix=evotune_sr_bpg0 \
    num_rounds=1000 \
    finetuning_frequency=200
```

### Running Different Problems

Override the problem via command line:

```bash
# Physics Oscillator problem PO0
python src/experiments/main.py \
    task=sr \
    task.dataset_category=phys_osc \
    task.problem_name=PO0 \
    model=llama32 \
    train=dpo \
    prefix=evotune_sr_po0 \
    num_rounds=1000

# Chemistry Reaction problem CRK5
python src/experiments/main.py \
    task=sr \
    task.dataset_category=chem_react \
    task.problem_name=CRK5 \
    model=llama32 \
    train=dpo \
    prefix=evotune_sr_crk5 \
    num_rounds=1000

# Transformed Feynman equation (no OOD test available)
python src/experiments/main.py \
    task=sr \
    task.dataset_category=lsr_transform \
    task.problem_name=I.10.7_1_0 \
    model=llama32 \
    train=dpo \
    prefix=evotune_sr_feynman \
    num_rounds=1000
```

### Quick Test Run

```bash
python src/experiments/main.py \
    task=sr \
    model=llama32 \
    train=none \
    prefix=sr_quick_test \
    num_rounds=5 \
    num_prompts_per_round=2 \
    num_outputs_per_prompt=2 \
    wandb=0
```

## Testing the Implementation

Run these commands to verify the implementation:

```bash
# Activate environment
source /apps/python/3.10/etc/profile.d/conda.sh
conda activate evotune
export PYTHONPATH=src:$PYTHONPATH

# Test 1: Check task registration
python -c "
from packing.evaluate.registry import TASK_REGISTRY
task = TASK_REGISTRY.get('sr')
print('Task registered:', list(task.keys()))
"

# Test 2: Test data loading with correct mapping
python -c "
from omegaconf import OmegaConf
from packing.evaluate.symbolic_regression.task_sr import generate_input

cfg = OmegaConf.create({
    'dataset_category': 'bio_pop_growth',
    'problem_name': 'BPG0',
})

train = generate_input(cfg, 'train')
print(f'train -> {train[\"split\"]}, shape={train[\"X\"].shape}')

perturbed = generate_input(cfg, 'trainperturbedset')
print(f'trainperturbedset -> {perturbed[\"split\"]}, shape={perturbed[\"X\"].shape}')
"

# Test 3: Test final evaluation
python -c "
from omegaconf import OmegaConf
from packing.evaluate.symbolic_regression.task_sr import evaluate_best_program_all_splits

cfg = OmegaConf.create({
    'dataset_category': 'bio_pop_growth',
    'problem_name': 'BPG0',
    'function_str_to_extract': 'equation',
})

func = 'def equation(X): return X[:, 0] * 0.01'
results = evaluate_best_program_all_splits(cfg, func, 'import numpy as np')
print(f'Train NMSE: {results[\"train\"][\"nmse\"]:.4f}')
print(f'Test NMSE: {results[\"test\"][\"nmse\"]:.4f}')
print(f'OOD NMSE: {results[\"ood_test\"][\"nmse\"]:.4f}')
"
```

## File Tree

```
EvoTune/
├── configs/
│   └── task/
│       └── sr.yaml                          # NEW: SR task config
├── src/
│   ├── experiments/
│   │   └── main.py                          # MODIFIED: Added final SR evaluation
│   └── packing/
│       ├── evaluate/
│       │   └── symbolic_regression/         # NEW: SR task directory
│       │       └── task_sr.py               # NEW: SR task implementation
│       └── funsearch/
│           └── programs_database.py         # MODIFIED: Added get_best_program
├── llm-srbench-dataset/                     # DATA: Cloned dataset
│   └── llm-srbench/
│       └── lsr_bench_data.hdf5              # HDF5 data file (239 problems)
└── add_symbolic_regression_task.md          # NEW: This documentation
```

## Notes

1. **Auto-discovery**: The task is automatically discovered and registered because `src/packing/evaluate/__init__.py` uses `rglob("task_*.py")` to find all task files.

2. **No manual import needed**: The auto-discovery handles task registration automatically.

3. **Score interpretation**:
   - Score of 0 = perfect prediction (NMSE = 0)
   - Score of -100 = NMSE of 1 (predictions as good as mean)
   - More negative = worse predictions

4. **Memory considerations**: The lsr_transform problems have larger datasets (up to 80,000 samples), so you may need to increase `mem_limit_gb` for those.

5. **Problem difficulty**: The benchmark is designed to prevent memorization, so problems are challenging. Expect initial scores to be quite negative.

6. **OOD availability**: Out-of-distribution test sets are only available for `lsr_synth` problems (bio_pop_growth, chem_react, phys_osc, matsci). The `lsr_transform` problems do not have OOD test sets.

7. **Wandb logging**: Final metrics are automatically logged to wandb with keys like `final/train_nmse`, `final/test_nmse`, `final/ood_nmse`.

## Problem Browser

A problem browser utility is available to explore and select problems:

```bash
# Activate environment
source /apps/python/3.10/etc/profile.d/conda.sh
conda activate evotune

# List all categories with summaries
python src/packing/evaluate/symbolic_regression/problem_browser.py --list-categories

# List problems in a category (sorted by complexity)
python src/packing/evaluate/symbolic_regression/problem_browser.py --category bio_pop_growth --sort-by complexity

# Show details for a specific problem (including ground truth equation)
python src/packing/evaluate/symbolic_regression/problem_browser.py --problem BPG10

# Show the N simplest problems across all categories
python src/packing/evaluate/symbolic_regression/problem_browser.py --simplest 15

# Filter by number of input variables
python src/packing/evaluate/symbolic_regression/problem_browser.py --num-inputs 2

# Interactive mode
python src/packing/evaluate/symbolic_regression/problem_browser.py --interactive
```

### Recommended Starting Problems

**Simplest problems (by complexity score):**

| Problem | Category | Inputs | Ground Truth |
|---------|----------|--------|--------------|
| BPG10 | bio_pop_growth | 2 | `0.101*P^(1/3) + 0.101*P` |
| CRK5 | chem_react | 2 | `-0.325*√A + 0.325*A^(1/3)` |
| CRK16 | chem_react | 2 | `-0.246*A + 0.246*A^(1/3)` |
| II.27.18_1_0 | lsr_transform | 2 | `-sqrt(E_den/epsilon)` |

**More challenging (multiple inputs, complex equations):**

| Problem | Category | Inputs | Description |
|---------|----------|--------|-------------|
| MatSci0 | matsci | 2 | Stress vs strain and temperature |
| PO0 | phys_osc | 3 | Nonlinear oscillator with position, time, velocity |

## References

- [LLM-SRBench Paper](https://arxiv.org/abs/2504.10415) (ICML 2025 Oral)
- [LLM-SRBench GitHub](https://github.com/deep-symbolic-mathematics/llm-srbench)
- [LLM-SRBench Dataset](https://huggingface.co/datasets/nnheui/llm-srbench)

How to Customize EvoTune for Your Needs

  Based on your current setup with osc_evotune.sh, here's exactly how to customize the repository:

  ---
  1. Adding New Models (Qwen 3, Gemma)

  Step 1.1: Create Model Config Files

  Create new YAML files in configs/model/:

  For Qwen 3 (configs/model/qwen3.yaml):
  model_name: "qwen3"
  temperature: 0.7
  topk: 100
  topp: 0.95
  max_tokens: 2048

  For Gemma (configs/model/gemma.yaml):
  model_name: "gemma"
  temperature: 0.7
  topk: 100
  topp: 0.95
  max_tokens: 2048

  Step 1.2: Add Model IDs to get_full_model_name()

  Edit src/packing/model/model.py:37-52 to add your new models:

  def get_full_model_name(cfg):
      def get_name(name):
          if name == "granite":
              model_id = "ibm-granite/granite-3.1-2b-instruct"
          elif name == "llama32":
              model_id = "meta-llama/Llama-3.2-1B-Instruct"
          elif name == "phi":
              model_id = "microsoft/Phi-3.5-mini-instruct"
          elif name == "qwen3":
              model_id = "Qwen/Qwen2.5-1.5B-Instruct"  # or whichever Qwen3 model you want
          elif name == "gemma":
              model_id = "google/gemma-2-2b-it"  # or "google/gemma-2-9b-it"
          else:
              raise ValueError(f"Invalid model name: {name}")
          return model_id

  Popular Model IDs to choose from:
  - Qwen 3: Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct, Qwen/Qwen2.5-7B-Instruct
  - Gemma: google/gemma-2-2b-it, google/gemma-2-9b-it, google/gemma-2-27b-it

  Step 1.3: Update Your Launch Script

  Modify your osc_evotune.sh (line 45):

  # Change from:
  model=llama32

  # To:
  model=qwen3
  # or
  model=gemma

  That's it for models! The rest of the system automatically handles:
  - vLLM server startup with the correct HuggingFace model ID
  - Tokenizer loading
  - LoRA fine-tuning configuration

  ---

● 2. Adding Custom Tasks

  The task system uses a registry pattern. Here's the complete process:

  Step 2.1: Create Task Directory and Implementation

  Create a new directory: src/packing/evaluate/your_task/

  Then create src/packing/evaluate/your_task/task_your_task.py:

  import numpy as np
  from omegaconf import DictConfig
  from packing.evaluate.registry import TASK_REGISTRY
  from packing.logging.function_class import FunctionClass
  import traceback

  # ============================================
  # 1. DEFINE INITIAL HEURISTIC
  # ============================================
  def get_initial_func(cfg):
      """
      Returns the initial/baseline function to seed the evolutionary search.
      
      Returns:
          tuple: (function_object, "function_name_as_string")
      """
      def your_heuristic_function(input_data):
          """Describe what this function does.
          
          Args:
              input_data: Description of input
              
          Returns:
              Description of output
          """
          # Your baseline algorithm here
          return result

      return your_heuristic_function, "your_heuristic_function"


  # ============================================
  # 2. GENERATE INPUT DATA
  # ============================================
  def generate_input(cfg, set: str):
      """
      Generate or load the dataset for evaluation.
      
      Args:
          cfg: Hydra config
          set: One of "train", "trainperturbedset", "testset"
          
      Returns:
          Your dataset (can be dict, list, object, etc.)
      """
      if set == "train":
          # Return training data
          return your_training_dataset
      elif set == "trainperturbedset":
          # Return perturbed/augmented training data
          return your_perturbed_dataset
      elif set == "testset":
          # Return test/validation data
          return your_test_dataset
      else:
          raise ValueError(f"Invalid dataset set: {set}")


  # ============================================
  # 3. EVALUATE FUNCTION
  # ============================================
  def evaluate_func(cfg, dataset, function_class: FunctionClass) -> FunctionClass:
      """
      Evaluate a generated function on your task.
      
      Args:
          cfg: Hydra config
          dataset: Output from generate_input()
          function_class: FunctionClass containing the generated function
          
      Returns:
          FunctionClass with updated score and eval fields
      """
      try:
          # Extract the function from function_class
          func = function_class.function

          # Run your evaluation logic
          total_score = 0
          for instance in dataset:
              result = func(instance)
              score = compute_score(result, instance)
              total_score += score

          # Average score or whatever metric you want to MAXIMIZE
          final_score = total_score / len(dataset)

          # Update function_class with results
          function_class.score = final_score
          function_class.fail_flag = 0

          # Optional: Store task-specific evaluation data
          function_class.eval = {
              'metric1': some_value,
              'metric2': another_value,
          }

      except Exception as e:
          # Handle evaluation failure
          function_class.score = cfg.failed_score
          function_class.fail_flag = 1
          function_class.eval = {'error': str(e), 'traceback': traceback.format_exc()}

      return function_class


  # ============================================
  # 4. REGISTER THE TASK
  # ============================================
  TASK_REGISTRY.register(
      "your_task",  # Task name used in command line
      generate_input=generate_input,
      evaluate_func=evaluate_func,
      get_initial_func=get_initial_func,
      system_prompt="""You are an expert algorithm designer. 
  Your goal is to discover a new heuristic function that solves [describe your problem].

  The function signature is:
  def your_heuristic_function(input_data):
      # Your code here
      return result

  The objective is to [maximize/minimize] [your metric].
  """,
      append_prompt="""Please provide an improved version of the function that achieves better performance."""
  )

  Step 2.2: Create Task Config File

  Create configs/task/your_task.yaml:

  # Task your_task
  task_name: "your_task"
  function_str_to_extract: "your_heuristic_function"  # Name of the function to extract

  # Evaluation settings
  failed_score: -99999  # Score assigned to failed/invalid functions
  timeout_period: 120   # Max seconds per function evaluation
  mem_limit_gb: 10      # Memory limit for evaluation workers

  # ProgramsDatabase settings
  programdatabaseConfig:
    temp: 40.0           # Temperature for island sampling
  initial_percentile: 0.3  # Starting quality percentile for sampling

  Step 2.3: Ensure Task is Auto-Loaded

  Check src/packing/evaluate/__init__.py - it should have:

  def import_all_tasks():
      from packing.evaluate.bin_packing import task_bin
      from packing.evaluate.tsp import task_tsp
      from packing.evaluate.flat_pack import task_flat_pack
      from packing.evaluate.your_task import task_your_task  # Add this line

  If the file doesn't exist or doesn't have this pattern, you may need to create it or manually import your task.

  Step 2.4: Update Launch Script

  Modify osc_evotune.sh (line 44):

  # Change from:
  task=bin

  # To:
  task=your_task

  ---

● 3. Example Commands & Best Practices

  Running Experiments with Different Models

  Test Qwen 3:
  sbatch scripts/osc_evotune.sh \
      --export=MODEL=qwen3,PREFIX=evotune_qwen3_1B

  Or edit osc_evotune.sh directly:
  python src/experiments/main.py \
      task=bin \
      model=qwen3 \           # Changed
      train=dpo \
      cluster=osc_ascend \
      prefix=evotune_qwen3 \  # Changed
      num_rounds=2701 \
      finetuning_frequency=400

  Compare Multiple Models (FunSearch baseline - no fine-tuning):
  # Gemma baseline
  python src/experiments/main.py \
      task=bin \
      model=gemma \
      train=none \              # No DPO training
      prefix=funsearch_gemma \
      num_rounds=2701

  Running Custom Task Experiments

  python src/experiments/main.py \
      task=your_task \           # Your custom task
      model=qwen3 \
      train=dpo \
      prefix=your_task_qwen3 \
      num_rounds=1000 \
      finetuning_frequency=200

  Key Configuration Parameters

  Performance tuning:
  num_workers=12              # Parallel evaluation workers (match cpus-per-task)
  num_outputs_per_prompt=4    # LLM generations per prompt
  num_prompts_per_round=32    # Prompts per round

  Training schedule:
  num_rounds=2701             # Total evolutionary rounds
  num_cont_rounds=100         # Rounds before checkpoint
  finetuning_frequency=400    # Trigger DPO every N rounds

  Evolutionary search:
  initial_percentile=0.3      # Start sampling from top 30%
  final_percentile=0.1        # End sampling from top 10%
  num_islands=6               # Number of independent populations

  ---
  4. File Checklist for New Models/Tasks

  For New Models:

  - configs/model/your_model.yaml - Created with model_name, temperature, etc.
  - src/packing/model/model.py - Updated get_full_model_name() with HuggingFace ID

  For New Tasks:

  - src/packing/evaluate/your_task/task_your_task.py - Implemented 3 functions + registration
  - configs/task/your_task.yaml - Created with task settings
  - src/packing/evaluate/__init__.py - Added import (if needed)

  ---
  5. Testing Your Changes

  Quick test run (5 rounds, no training):
  python src/experiments/main.py \
      task=your_task \
      model=qwen3 \
      train=none \
      prefix=quick_test \
      num_rounds=5 \
      num_prompts_per_round=2 \
      num_outputs_per_prompt=2 \
      wandb=0

  Check outputs:
  - Logs: out/logs/quick_test/*/stdout.log
  - Scores: out/logs/quick_test/*/programdb_scores.json
  - Generated functions: out/logs/quick_test/*/best_programs.txt

  ---
  6. Common Issues & Solutions

  Issue: "Invalid model name" error
  - Solution: Check that your model name in config file matches the case in get_full_model_name()

  Issue: Task not found
  - Solution: Ensure import_all_tasks() imports your task module, or manually import it in __init__.py

  Issue: vLLM fails to start
  - Solution: Check HuggingFace model ID is correct and accessible. May need export HF_TOKEN=... for gated models (Llama, Gemma)

  Issue: Memory errors during evaluation
  - Solution: Reduce num_workers or increase mem_limit_gb in task config

  ---

● Summary

  To customize EvoTune for your experiments:

  Adding Models (Qwen 3, Gemma):

  1. Create configs/model/qwen3.yaml and configs/model/gemma.yaml
  2. Edit src/packing/model/model.py:37-52 to add model IDs
  3. Change model=qwen3 in your launch script

  Adding Tasks:

  1. Create src/packing/evaluate/your_task/task_your_task.py with:
    - get_initial_func() - baseline heuristic
    - generate_input() - dataset loader
    - evaluate_func() - scoring logic
    - TASK_REGISTRY.register() - registration
  2. Create configs/task/your_task.yaml with task settings
  3. Import task in src/packing/evaluate/__init__.py
  4. Change task=your_task in launch script

  Both changes are modular - you don't need to modify core evolutionary or training logic. The registry pattern handles everything automatically once registered.

  ---
  Would you like me to:
  1. Create the Qwen 3 and Gemma model configs for you?
  2. Show you a specific example task implementation (e.g., for a knapsack problem, scheduling task, etc.)?
  3. Help you implement a specific custom task you have in mind?

  Let me know what you'd like to tackle next!
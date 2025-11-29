#!/bin/bash
#SBATCH --job-name=funsearch_sr_BPG10
#SBATCH --account=PAA0201
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --partition=quad
#SBATCH --mem=100G
#SBATCH --exclusive
#SBATCH --output=logs/funsearch_sr_%j.out
#SBATCH --error=logs/funsearch_sr_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=moussa.45@osu.edu

# ============================================
# Symbolic Regression - FunSearch Baseline
# ============================================
# This script runs the FunSearch baseline (no fine-tuning) on SR tasks.
#
# To change the problem, modify DATASET_CATEGORY and PROBLEM_NAME below.
# Use the problem browser to find problems:
#   python src/packing/evaluate/symbolic_regression/problem_browser.py --simplest 15
#   python src/packing/evaluate/symbolic_regression/problem_browser.py --problem BPG10
#
# Available categories: bio_pop_growth, chem_react, matsci, phys_osc, lsr_transform
# ============================================

# Problem configuration - CHANGE THESE TO RUN DIFFERENT PROBLEMS
DATASET_CATEGORY="bio_pop_growth"
PROBLEM_NAME="BPG10"

# Model configuration
MODEL="phi"  # Options: llama32, qwen3, etc.

# Run configuration
SEED=0
NUM_ROUNDS=2701
NUM_CONT_ROUNDS=100

# ============================================

# Print job information
echo "=========================================="
echo "Symbolic Regression - FunSearch Baseline"
echo "=========================================="
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo ""
echo "Problem: ${DATASET_CATEGORY}/${PROBLEM_NAME}"
echo "Model: ${MODEL}"
echo "=========================================="

# Load necessary modules
module load python/3.10 || module load python
module load cuda/12.8.1 || module load cuda/12.6.2 || module load cuda/12.4.1 || module load cuda/11.8.0 || module load cuda

# Initialize and activate conda environment
source /apps/python/3.10/etc/profile.d/conda.sh
conda activate evotune

# Set environment variables
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="faafa38294bc6098fd6475997f551a8de1ade862"

# Create logs directory if it doesn't exist
mkdir -p logs out/logs

# Construct prefix from problem name
PREFIX="funsearch_sr_${PROBLEM_NAME}_${MODEL}"

# Run the experiment
python src/experiments/main.py \
    task=sr \
    task.dataset_category=${DATASET_CATEGORY} \
    task.problem_name=${PROBLEM_NAME} \
    model=${MODEL} \
    train=none \
    cluster=osc_ascend \
    seed=${SEED} \
    prefix=${PREFIX} \
    num_rounds=${NUM_ROUNDS} \
    num_cont_rounds=${NUM_CONT_ROUNDS} \
    gpu_nums=0 \
    wandb=1 \
    project=cse-6521-project \
    entity=hananenmoussa \
    run_or_dev=run

echo ""
echo "=========================================="
echo "Job finished on $(date)"
echo "Results saved to: out/logs/${PREFIX}/"
echo "Final metrics: out/logs/${PREFIX}/*/metrics/final_sr_metrics.json"
echo "=========================================="

#!/bin/bash
#SBATCH --job-name=funsearch_qwen3_no_think_1.7B_baseline
#SBATCH --account=PAA0201
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=1
#SBATCH --partition=quad
#SBATCH --mem=100G
#SBATCH --output=logs/funsearch_qwen3_no_think_1.7B_baseline_%j.out
#SBATCH --error=logs/funsearch_qwen3_no_think_1.7B_baseline_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=moussa.45@osu.edu

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

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

python src/experiments/main.py \
    task=bin \
    model=qwen3 \
    train=none \
    cluster=osc_ascend \
    seed=0 \
    prefix=funsearch_qwen3_no_think_1.7B_baseline \
    num_rounds=2701 \
    num_cont_rounds=100 \
    gpu_nums=0 \
    wandb=1 \
    project=cse-6521-project \
    entity=hananenmoussa \
    run_or_dev=run

echo "Job finished on $(date)"

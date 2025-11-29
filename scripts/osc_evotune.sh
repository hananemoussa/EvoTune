#!/bin/bash
#SBATCH --job-name=evotune_qwen3_no_think_1.7B 
#SBATCH --account=PAA0201    # TODO: Replace with your OSC project allocation
#SBATCH --time=72:00:00                 # 48 hours (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12              # Matches num_workers in config
#SBATCH --gpus-per-node=1               # Request 1 GPU for single run
#SBATCH --partition=quad                # Options: nextgen (2 GPUs), quad (4 GPUs), batch (4 GPUs)
#SBATCH --mem=100G                      # Memory request
#SBATCH --output=logs/evotune_qwen3_no_think_1.7B_%j.out
#SBATCH --error=logs/evotune_qwen3_no_think_1.7B_%j.err
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

# Alternative: If using Apptainer/Singularity container instead
# apptainer run --nv <your_docker_image.sif> ...

# Set environment variables
export PYTHONPATH=src:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY="faafa38294bc6098fd6475997f551a8de1ade862"

# Create logs directory if it doesn't exist
mkdir -p logs out/logs

# Run the training
python src/experiments/main.py \
    task=bin \
    model=qwen3 \
    train=dpo \
    cluster=osc_ascend \
    seed=0 \
    prefix=evotune_qwen3_no_think_1.7B \
    gpu_nums=0 \
    num_rounds=2701 \
    num_cont_rounds=100 \
    finetuning_frequency=400 \
    one_tuning=1 \
    max_loops=1 \
    wandb=1 \
    project=cse-6521-project \
    entity=hananenmoussa \
    run_or_dev=run


echo "Job finished on $(date)"



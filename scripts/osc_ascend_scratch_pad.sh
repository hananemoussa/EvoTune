#!/bin/bash
#SBATCH --job-name=evotune_llama_1B 
#SBATCH --account=PAA0201    # TODO: Replace with your OSC project allocation
#SBATCH --time=72:00:00                 # 48 hours (adjust as needed)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12              # Matches num_workers in config
#SBATCH --gpus-per-node=1               # Request 1 GPU for single run
#SBATCH --partition=quad                # Options: nextgen (2 GPUs), quad (4 GPUs), batch (4 GPUs)
#SBATCH --mem=100G                      # Memory request
#SBATCH --output=logs/evotune_llama_1B_%j.out
#SBATCH --error=logs/evotune_llama_1B_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=moussa.45@osu.edu

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"

# Load necessary modules
# Note: Adjust versions based on what's available on your OSC system
# Use 'module avail python' and 'module avail cuda' to see options
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
    model=llama32 \
    train=dpo \
    cluster=osc_ascend \
    seed=0 \
    prefix=evotune_llama_1B_test \
    gpu_nums=0 \
    num_rounds=20 \
    num_cont_rounds=5 \
    finetuning_frequency=10 \
    one_tuning=0 \
    max_loops=5 \
    wandb=1 \
    project=evotune-reproducing \
    entity=hananenmoussa \
    run_or_dev=run


echo "Job finished on $(date)"


# My workflow: 
# Request gpu node interactively using: sinteractive -A PAA0201 -p quad -g 1 -t 1:00:00
# Then: conda activate evotune
# Then run the command python src/experiments/main.py \ ...

# ASCEND USEFUL COMMANDS:

# Your Jobs

#   squeue -u $USER                    # Your jobs on current cluster
#   squeue --cluster=all -u $USER      # Your jobs on all clusters

#   All Jobs in Queue (Ascend cluster)

#   squeue -p quad                     # All jobs in quad partition
#   squeue -p quad | wc -l             # Count jobs in queue
#   squeue --partition=quad,nextgen    # Multiple partitions

#   Detailed View

#   squeue --Format=jobid,partition,name,username,state,timeLeft,numCPUS,numNodes

#   System-Wide Status

#   - OnDemand Portal: Go to Clusters → System Status
#   - Shows: nodes in use, cores in use, running/queued/blocked jobs

#   Estimate Start Time

#   squeue --start -j <job_id>         # Estimated start time for your job


# Useful Commands for OSC Ascend

#   1. Check Your Jobs

#   # Basic check - your jobs only
#   squeue -u $USER

#   # Detailed view with more columns
#   squeue -u $USER -o "%.18i %.12P %.30j %.8u %.2t %.10M %.6D %R"

#   # Check all jobs on a specific partition (e.g., gpu-ascend)
#   squeue -p gpu-ascend

#   2. Check Recent Job History

#   # Your completed jobs from today
#   sacct -u $USER --starttime today --format=JobID,JobName,Partition,State,Elapsed,ExitCode

#   # Last 7 days
#   sacct -u $USER --starttime $(date -d '7 days ago' +%Y-%m-%d) --format=JobID,JobName,Partition,State,Elapsed,ExitCode

#   3. Check Partition Info

#   # See available partitions on Ascend
#   sinfo

#   # Check GPU partition specifically
#   sinfo -p gpu-ascend

#   4. Check Your Account/Allocation

#   # Check your project allocation and usage
#   sbalance -u $USER

#   # Or check all accounts you have access to
#   sacctmgr show associations user=$USER -p

#   ---
#   Common OSC Ascend Partitions

#   Based on OSC documentation, typical Ascend partitions include:
#   - gpu-ascend - Main GPU partition
#   - ascend - General compute
#   - Check with sinfo to see what's available

#   ---

# Monitoring commands: 
# 1. Quick Status Check

#   squeue -u $USER

#   2. Watch Live Progress (updates every 5 seconds)

#   watch -n 5 'squeue -u $USER -o "%.10i %.12P %.30j %.2t %.10M %R"'
#   # Press Ctrl+C to exit

#   3. View Real-Time Logs

#   Baseline job:
#   tail -f logs/funsearch_llama_1B_baseline_2916473.out

#   EvoTune job:
#   tail -f logs/evotune_llama_1B_2916475.out

#   4. Check Both Logs in Split Screen

#   # Terminal 1:
#   tail -f logs/funsearch_llama_1B_baseline_2916473.out

#   # Terminal 2:
#   tail -f logs/evotune_llama_1B_2916475.out

#   5. Check for Errors

#   # Baseline errors
#   tail -50 logs/funsearch_llama_1B_baseline_2916473.err

#   # EvoTune errors
#   tail -50 logs/evotune_llama_1B_2916475.err

#   6. View Detailed Job Info

#   scontrol show job 2916473  # Baseline
#   scontrol show job 2916475  # EvoTune

# ---
#   🛠️ Useful Management Commands

#   Cancel a Job (if needed)

#   scancel 2916473  # Cancel baseline
#   scancel 2916475  # Cancel evotune
#   # Or cancel both:
#   scancel 2916473 2916475

#   Check Job History

#   sacct -j 2916473,2916475 --format=JobID,JobName,State,Elapsed,ExitCode

#   Check Resource Usage

#   seff 2916473  # After job completes
#   seff 2916475
#!/bin/bash
#SBATCH --job-name=parallel_launcher
#SBATCH --partition=regular1,regular2
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.err

# Create logs directory if it does not exist
mkdir -p ./logs
echo $(pwd)
# Define arrays of parameters

alphas=(0.04)
Ts=($(seq 0.05 0.05 1.60))

# Timestamp for the launch
echo "Launcher started at $(date)"
echo "Submitting jobs for all combinations of N and T"

# Loop over all combinations
for alpha in "${alphas[@]}"; do
    for T in "${Ts[@]}"; do
        # Construct a unique job name and log files
        JOB_NAME="T${T}"
        OUT_LOG="./logs/job_${JOB_NAME}.out"
        ERR_LOG="./logs/job_${JOB_NAME}.err"

        echo "Submitting job for alpha=$alpha T=$T at $(date)"
        sbatch --job-name="$JOB_NAME" \
               --output="$OUT_LOG" \
               --error="$ERR_LOG" \
               ./shell/launch_job.sh "$alpha" "$T"

        # Small delay to avoid overwhelming the scheduler
        sleep 0.01
    done
done

echo "All jobs submitted at $(date)"

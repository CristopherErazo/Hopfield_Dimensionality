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

# Ns=(4096 8192 16384 32768)
# Ts=($(seq 0.80 0.02 1.60))

Ns=(1024)
Ts=(1.2)

# Timestamp for the launch
echo "Launcher started at $(date)"
echo "Submitting jobs for all combinations of N and T"

# Loop over all combinations
for T in "${Ts[@]}"; do
    for N in "${Ns[@]}"; do
        # Construct a unique job name and log files
        JOB_NAME="N${N}"
        OUT_LOG="./logs/job_${JOB_NAME}.out"
        ERR_LOG="./logs/job_${JOB_NAME}.err"

        echo "Submitting job for alpha=$alpha at $(date)"
        sbatch --job-name="$JOB_NAME" \
                --output="$OUT_LOG" \
                --error="$ERR_LOG" \
                ./shell/launch_job.sh "$N" "$T"

        # Small delay to avoid overwhelming the scheduler
        sleep 0.01
    done
done

echo "All jobs submitted at $(date)"

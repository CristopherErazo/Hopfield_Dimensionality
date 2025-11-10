#!/bin/bash
#SBATCH --job-name=parallel_launcher
#SBATCH --partition=regular1,regular2
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=../logs/launcher-%j.out
#SBATCH --error=../logs/launcher-%j.err

# Create logs directory if it does not exist
mkdir -p ../logs

# Define arrays of parameters
Ns=(1024 2048 4096)
Ts=(0.6 0.9 1.2 1.5)

# Timestamp for the launch
echo "Launcher started at $(date)"
echo "Submitting jobs for all combinations of N and T"

# Loop over all combinations
for N in "${Ns[@]}"; do
    for T in "${Ts[@]}"; do
        # Construct a unique job name and log files
        JOB_NAME="N${N}_T${T}"
        OUT_LOG="../logs/job_${JOB_NAME}_%j.out"
        ERR_LOG="../logs/job_${JOB_NAME}_%j.err"

        echo "Submitting job for N=$N T=$T at $(date)"
        sbatch --job-name="$JOB_NAME" \
               --output="$OUT_LOG" \
               --error="$ERR_LOG" \
               jobscript.sh "$N" "$T"

        # Small delay to avoid overwhelming the scheduler
        sleep 0.01
    done
done

echo "All jobs submitted at $(date)"

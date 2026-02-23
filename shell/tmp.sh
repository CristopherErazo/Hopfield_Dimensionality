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

# Loop over all combinationsi

configurations=(
'4096 1.06 19'
'4096 1 19'
'8192 1.08 19'
'4096 1.02 16'
'16384 1.22 13'
'4096 1.36 13'
'4096 1.08 12'
'4096 1.1 12'
'4096 1.32 12'
'4096 1.38 12'
'4096 1.24 11'
'8192 1.4 11'
'4096 1.12 10'
'4096 1.3 10'
'8192 1.38 10'
'4096 1.26 9'
'4096 1.2 9'
'4096 1.4 9'
'4096 1.14 8'
'4096 1.16 8'
'4096 1.18 8'
'4096 1.22 8'
'4096 1.28 8'
'8192 1.02 8'
'8192 1.06 8'
'8192 1.34 7'
'8192 1.3 7'
'16384 1.28 6'
'8192 1 6'
'4096 1.34 5'
'8192 1.24 5'
'8192 1.36 5'
'16384 1.04 4'
'8192 1.04 4'
'16384 1.08 3'
'8192 1.26 3'
'8192 1.32 3'
'16384 1.34 2'
'16384 1.36 2'
'8192 1.14 2'
'8192 1.16 2'
'8192 1.22 2'
'8192 1.28 2'
'16384 1.02 1'
'16384 1 1'
'8192 1.12 1'
'8192 1.18 1'
'8192 1.1 1'

)


for config in "${configurations[@]}"; do
    read -r  N T Nit <<< "$config"

        # Construct a unique job name and log files
        JOB_NAME="Nit${Nit}"
        OUT_LOG="./logs/job_${JOB_NAME}.out"
        ERR_LOG="./logs/job_${JOB_NAME}.err"

        echo "Submitting job for alpha=$alpha at $(date)"
        sbatch --job-name="$JOB_NAME" \
                --output="$OUT_LOG" \
                --error="$ERR_LOG" \
                ./shell/launch_job.sh "$N" "$T" "$Nit"

        # Small delay to avoid overwhelming the scheduler
	sleep 0.01
done

echo "All jobs submitted at $(date)"
                                         

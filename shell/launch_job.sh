#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=00:30:00
#SBATCH --mem=10G
#SBATCH --partition=regular1,regular2
# SBATCH --qos=fastlane # for debugging

mkdir -p ./logs

module load gnu8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hop_bid

echo $(pwd)
# Set environment variables for JAX and threading
JOBS=8
THREADS_PER_JOB=$(( SLURM_CPUS_PER_TASK / JOBS ))
export OMP_NUM_THREADS=$THREADS_PER_JOB

export OMP_NUM_THREADS=$THREADS_PER_JOB
# export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_multi_thread_eigen_thread_count=$THREADS_PER_JOB"

# Slurm job information
echo "SLURM job info:"
echo "  SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "  SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "  SLURM_NTASKS = $SLURM_NTASKS"
echo "  SLURM_JOB_CPUS_PER_NODE = $SLURM_JOB_CPUS_PER_NODE"

# Define parsed parameters
alpha=$1
T=$2

# Define fixed parameters
N=1024
N_samples=2500
burnin=1500
progress=False

# Define variable parameters
Nw=60

start_time=$(date +%s)
echo "Starting job $SLURM_JOB_ID at $(date)"


for h in 0.0 0.9; do  
    export N T alpha h N_samples burnin progress SLURM_JOB_ID

    # Define a single log file per job
    log_file="./logs/job_alpha${alpha}_T${T}_h${h}.log"


    seq 1 $Nw | parallel -j $JOBS '
    echo "=== Running iteration {} for job $SLURM_JOB_ID ===" >> '"$log_file"'
    python -u ./scripts/run_cuts.py --N $N --T $T --alpha $alpha --h $h \
        --N_samples $N_samples --progress $progress \
        --burnin $burnin --iteration {} \
        >> '"$log_file"' 2>&1
    '

    end_time=$(date +%s)
    elapsed=$(( end_time - start_time ))
    echo "Total time: ${elapsed} seconds = $(( elapsed / 60 )) min"
done

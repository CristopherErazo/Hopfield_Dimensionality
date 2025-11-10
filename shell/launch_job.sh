#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=12:00:00
#SBATCH --mem=20G
#SBATCH --partition=regular1,regular2
# SBATCH --qos=fastlane # for debugging

mkdir -p ../logs

source .venv/bin/activate

# Set environment variables for JAX and threading
JOBS=4
THREADS_PER_JOB=$(( SLURM_CPUS_PER_TASK / JOBS ))
export OMP_NUM_THREADS=$THREADS_PER_JOB

export OMP_NUM_THREADS=$THREADS_PER_JOB
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true --xla_cpu_multi_thread_eigen_thread_count=$THREADS_PER_JOB"

# Slurm job information
echo "SLURM job info:"
echo "  SLURM_CPUS_ON_NODE = $SLURM_CPUS_ON_NODE"
echo "  SLURM_CPUS_PER_TASK = $SLURM_CPUS_PER_TASK"
echo "  SLURM_NTASKS = $SLURM_NTASKS"
echo "  SLURM_JOB_CPUS_PER_NODE = $SLURM_JOB_CPUS_PER_NODE"

# Define parsed parameters
N=$1
T=$2

# Define fixed parameters
alpha=0.04
h=0.9
N_samples=10
progress=False

# Define variable parameters
Nw=4

start_time=$(date +%s)
echo "Starting job $SLURM_JOB_ID at $(date)"


export N T alpha h N_samples progress SLURM_JOB_ID

seq 1 $Nw | parallel -j $JOBS '
echo "Running iteration {} for job $SLURM_JOB_ID"
python -u ./scripts/run_scaling.py --N $N --T $T --alpha $alpha --h $h \
    --N_samples $N_samples --progress $progress \
    > ../logs/job_N${N}_T${T}_${SLURM_JOB_ID}_run_{}.log 2>&1
'

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time: ${elapsed} seconds = $(( elapsed / 60 )) minutes"


# for it in $(seq 1 1 $Nw); do
#     start_iter=$(date +%s)
#     echo "------------    RUNNING iteration $it / $Nw    ------------"
#     echo "Starting iteration at $(date)"
#     python -u ./scripts/run_scaling.py --N $N --T $T --alpha $alpha --h $h \
#     --N_samples $N_samples --progress $progress
#     end_iter=$(date +%s)
#     elapsed_iter=$(( end_iter - start_iter ))
#     echo "Iteration completed at $(date), took ${elapsed_iter} seconds = $(( elapsed_iter / 60 )) minutes"
#     echo "----------------------------------------"
#     echo ""
# done
#!/bin/bash

#SBATCH --job-name=filtering
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --mem=5G
#SBATCH --partition=regular1,regular2
# SBATCH --qos=fastlane # for debugging
#SBATCH --output=logs/job-%j.out
#SBATCH --error=logs/job-%j.err


mkdir -p ./logs

module load gnu8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hop_bid

echo $(pwd)

z_th=1.5
perc=80.0
test_modes=False

echo "Starting filtering with z_th=$z_th and perc=$perc at $(date)"
python -u ./scripts/analyze_cuts.py --z_th $z_th --perc $perc --burnin 1500 --N_iterations 60 --test_modes $test_modes
echo "Filtering completed at $(date)"

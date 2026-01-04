#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=05:00:00
#SBATCH --mem=5G
#SBATCH --partition=regular1,regular2
# SBATCH --qos=fastlane # for debugging

mkdir -p ./logs

module load gnu8
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hop_bid

echo $(pwd)

z_th=2.0
perc=80.0


echo "Starting filtering with z_th=$z_th and perc=$perc at $(date)"
python -u ./scripts/analyze_cuts.py --z_th $z_th --perc $perc
echo "Filtering completed at $(date)"

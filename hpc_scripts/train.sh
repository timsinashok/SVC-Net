#!/bin/bash

#SBATCH --time=40:00:00
#SBATCH --partition=nvidia
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --ntasks=8
#SBATCH --mail-type=END
#SBATCH --mail-user=<NetID>@nyu.edu
#SBATCH --output=/scratch/netid/Lab-PI/october/outputs/5n-train.out

# # Activate conda environment
source /share/apps/NYUAD5/miniconda/3-4.11.0/bin/activate
module load cuda
conda activate /scratch/netid/scratch/netid/tf-gpu

# Script to Execute
python3 /scratch/netid/Lab-PI/OCTA500/SVC-Net/train2.py
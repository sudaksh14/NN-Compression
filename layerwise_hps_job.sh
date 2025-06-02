#!/bin/bash
#Set job requirements
#SBATCH -J hyperparameter_search_layerwise
#SBATCH -t 48:00:00
#SBATCH -p gpu_mig
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --output=out/it_all_went_wrong-%A.out
module load 2024
module load CUDA/12.6.0

cd $HOME/NN-Compression/

#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
uv run iterative_pruner.py 

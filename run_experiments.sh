#!/bin/bash

#SBATCH --job-name=gnn_exps
#SBATCH --time=1:00
#SBATCH --mail-type=ALL
#SBATCH --output=gnn_exps.txt
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=24G 

module load miniconda
conda activate mooc

python gnn_experiments.py "$1"
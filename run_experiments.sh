#!/bin/bash

#SBATCH --job-name=gnn_exps_2
#SBATCH --time=4:00:00
#SBATCH --mail-type=ALL
#SBATCH --output=gnn_exps_2.txt
#SBATCH --ntasks=1
#SBATCH --partition=gpu_devel
#SBATCH --gpus=1
#SBATCH --mem=24G 

module load miniconda
conda activate mooc

python gnn_experiments.py "$1"
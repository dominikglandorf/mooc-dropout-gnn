# Dropout prediction from MOOC interaction data with Graph Neural Networks

This repository is part of a course project for CPSC483 at Yale University. The goal is to use Graph Neural Networks to predict whether an interaction in an online course environment will be the last one of a user.

# Setup
To setup the environment, install conda and run `conda env create -f environment.yml`. To activate it, run `conda activate mooc`.

Execute the file `download_mooc.sh` to automatically download the MOOC dataset into the directory `data/act-mooc`.

For the second dataset, manually download the three .csv files of the junyi dataset from [Kaggle](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy/) into the directory `data/junyi`. You need a free Kaggle account for this.

# Preprocessing
Run all cells in the notebook `data_preprocessing.ipynb` to create the necessary .pt files that will be used in the experiments.

# Experiments

For the experiments based on the GNN, run the cells in the file `gnn_experiments.ipynb` to add results to the file `results.csv` that contains the training details and metrics.

For the experiments based on the TGN, run the cells in the file `tgn_experiments.ipynb` to add results to the file `results.csv` that contains the training details and metrics.
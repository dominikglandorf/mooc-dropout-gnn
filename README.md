# Dropout prediction from MOOC interaction data with Graph Neural Networks

This repository is part of the course project for CPSC483 at Yale University by Dominik Glandorf. The goal is to use Graph Neural Networks to predict whether an interaction in an online course environment will be the last one of a user.

# Setup
To setup the environment, install conda and run `conda env create -f environment.yml`. To activate it, run `conda activate mooc`.

Execute the file `download_mooc.sh` to automatically download the MOOC dataset into the directory `data/act-mooc`.

For the second dataset, manually download the three .csv files of the junyi dataset from [Kaggle](https://www.kaggle.com/datasets/junyiacademy/learning-activity-public-dataset-by-junyi-academy/) into the directory `data/junyi`. You need a free Kaggle account for this.

# Preprocessing
Run all cells in the notebook `data_preprocessing.ipynb` to create the necessary .pt files that will be used in the experiments.

# Experiments

For the experiments based on the GNN, execute the python script `gnn_experiments.py` or schedule it as a batch job via  `run_experiments.sh` to add results to the file `results.csv` that contains the training details and metrics.

For the experiments based on the TGN, execute the python script `tgn_experiments.py` to add results to the file `results_tgn.csv` that contains the training details and metrics.

One can postprocess the results with the notebook `process_results.ipynb` to create a latex table or other statistics.
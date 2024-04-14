# Directory Contents
The notebooks in this directory were run (using Jupyter notebook) on the Azure machine with the GPUs.
A note about wandb:
- Create user.
- When running, put the API key in the WANDB_API_KEY environment variable

## PreprocessData
The code in this notebook is based on the beginning of: train/nllb_train_he_en.py
It prepares the data for loading and testing the model. 
The output datasets are stored in disk to be used later during training.

## ReproduceCurrentState
Provides an easy way to load and test the Model (the continuation of PreprocessData) based on train/nllb_train_he_en.py 

## Evaluate the model
Predicate, run comet and save results to a file.

## ExportModel
Save the model to disk.



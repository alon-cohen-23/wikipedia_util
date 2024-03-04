# Prepare training data

The files in this directory are all related to preparing the various datasets used during training and verification of the model.
All dependencies for preparing the datasets is stored in dataset/requirements.txt

After installation, make sure to run the following Python code from a Python terminal *once* before running:
```
import nltk
nltk.download('punkt')
```
## inss

## subtitles

## Text files
We assume that the input is a collection of text files. 

## wikipedia
Download pages from Hebrew wikipedia.

## full_flow
Run a full flow for preparing a dataset. The input parameters steps specifies what steps should run.
If a step exists in the list then it will be run. Otherwise, it will be either be skipped or read from disk.
The steps are:
1. Read all input data into a dataframe and filter (sentences with 4-30 words, only in Hebrew)
2. Replace entities (identified with [ner](../ner/README.md)) from the entity database. This step hasn't been used.
3. Split the file into multiple Excel files, each with maximum 30K rows (the maximum size allowed by Google Translate)
4. Run Google translate (using Selenium)
5. Manual step reminding the user to copy the output (translated) files from the Downloads directory
6. Concatenate all translated files (back) into one dataframe
7. Organize the dataset columns so they are ready for [Preprocess](../notebooks/PreprocessData.ipynb). Notice that this step is already part of [nllb_train_he_en](../train/nllb_train_he_en.py) 

## df_process
A few helper functions used by full_flow

## gtranslate_selenium
Code for running Google Translate using Selenium

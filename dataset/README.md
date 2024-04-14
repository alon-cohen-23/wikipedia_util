# Prepare training data

The files in this directory are all related to preparing the various datasets used during training and verification of the model.
All dependencies for preparing the datasets is stored in dataset/requirements.txt

After installation, make sure to run the following Python code from a Python terminal *once* before running:
```
import nltk
nltk.download('punkt')
```

## wikipedia
Download pages from Hebrew wikipedia.

## inss
Download pages from INSS.
This is done as follows:
1. Creating a [links (links.txt)](links.txt) file that contains the link to each of the articles.
2. For each line in the [links](links.txt) - extract the article in the link.

Because of the way the INSS website is structured, retrieving the links is done by accessing to the articles page, scrolling down to the bottom of the page, wait a few seconds and then more articles appear. This must be done repeatedly until no more articles appear. Because of this, it is impossible to analyze the page with static packages such as `BeatifulSoup`, and  it is necessary to use dynamic tools such as `Selenium`, specifically with `headless=False` (trying to use `headless=True` will only return the first group of articles).

In addition, it is necessary to wait (`sleep`) for a few seconds in order to let additional articles to appear. Therefore, the creation of the link file is relatively slow (~1-2 hours).


## subtitles
1. Download the subtitle files in `srt` format. (This is the accepted format for subtitles).
2. Run [retrieve_text_from_subtitles.py](inss\retrieve_text_from_subtitles.py). 
Currently the code assumes the following folder structure:
- Main folder. 
- Inside it, a collection of folders in the format:
`<program_name>_S<season>_heb`
- Inside each such folder, a collection of files of the form:
`<program_name>.S<season>E<episode>*.srt`
(where the season and episode appear in a 2-digit format).

However, it is easy to change this by changing the regex in the code.

(The function `get_subtitles_from_dir` receives the main directory as input, the name of the program and the number of seasons, and goes through all the files in the subdirectories of this program. 
The function `get_subtitles` works on a single srt file).

## Text files
We assume that the input is a collection of text files. 

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

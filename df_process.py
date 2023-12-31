# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:28:54 2023

@author: family123
"""

import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from pathlib import Path
from gtranslate_selenium import google_translate_folder_of_excels


def filter_sentences_df(df):
    """

    Parameters
    ----------
    df : pandas dataframe.

    Returns
    -------
    He_filtered_df : filter the df to sentences with 4-30 words, that written only in Hebrew.

    """
    # filter the length of the sentences (between 4-30 words)
    df['word_count'] = df['HE_sentences'].apply(lambda x: count_words(x))
    len_filtered_df = df.query('word_count > 3 and word_count < 31').copy()

    # filter the df to only hebrew sentences
    len_filtered_df['sen_lang'] = len_filtered_df['HE_sentences'].apply(lambda x: detect_lang(x))
    He_filtered_df = len_filtered_df.query("sen_lang == 'he'")

    He_filtered_df = He_filtered_df[['title', 'HE_sentences']]
    return He_filtered_df


def count_words(cell_content):
    words = cell_content.split()
    return len(words)


def detect_lang(cell_content):
    try:
        return detect(cell_content)
    except LangDetectException:
        return 'unknown'


def split_df(df):
    """

    Parameters
    ----------
    df : pandas dataframe.

    Returns
    -------
    splits the df to multiple excel file, all under the he_tr_excel folder.
    The function us necessary because google translate can not work with files that are bigger than about 30000 rows.
    """
    copied_df = df.copy()
    copied_df = copied_df.drop(columns=['title'])

    for number in range(int(len(df) / 30000 + 1)):  # the max size for google translate
        if len(df) > 30000:
            part_df = copied_df.head(30000)
            copied_df = copied_df.drop(copied_df.index[:30000])
        else:
            part_df = copied_df

        part_df.to_excel(f'he_tr_excel/{number}_part_HE.xlsx', index=False)


def concat_dir_excels(folder_path):
    concat_df = pd.DataFrame()

    for file_path in Path(folder_path).glob("*.xlsx"):
        file_df = pd.read_excel(file_path)
        concat_df = pd.concat([concat_df, file_df], ignore_index=True)

    return concat_df


def create_concatenated_translated_df(he_folder_path, en_folder_path):
    """
    Parameters
    ----------
    he_folder_path : path to the folder that contains the splited Hebrew sentences df
    en_folder : path to the folder that contains the splited translated English sentences df

    Returns
    -------
    concatenated df that contains the data of all of the files from both folders, drop the duplicates,
    and has 2 columns: HE_sentences, EN_sentences
    """
    df = concat_dir_excels('tr_data/he_tr_excel')

    en_df = concat_dir_excels('tr_data/en_tr_excel')
    en_df = en_df.rename(columns={'HE_sentences': 'EN_sentences'})

    df['EN_sentences'] = en_df['EN_sentences']
    df = df.drop_duplicates(
        subset=['HE_sentences'])  # I translated the duplicates as well so I cant drop them in the beggining

    df = df.reset_index(drop=True)

    return df


if __name__ == '__main__':
    # the 5 steps of translating a df

    df = pd.read_parquet('path/to/df')  # step 1: read the df
    split_df(df)  # step 2: split the df for Google translate

    he_folder_path = 'he_tr_excel'  # splited_df saves the files in this folder
    google_translate_folder_of_excels('he_tr_excel')  # step 3: translate the splited files

    # step 4: move the translated files from the downloads to new folder (or keep the download folder clean from irelevent excel files).

    en_folder_path = 'en_folder_path(change to your own)'
    create_concatenated_translated_df(he_folder_path, en_folder_path)  # step 5: concat everithing together

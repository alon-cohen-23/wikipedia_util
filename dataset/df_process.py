# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:28:54 2023

@author: family123
"""
from pathlib import Path
import numpy as np
import pandas as pd

lang_char_ranges = {
  'ar' : [[0x600, 0x6ff]],
  'fa' : [[0x600, 0x6ff],[0x750,0x77f],[0x8a0,0x8ff],[0xfb50,0xfdff]],
  'he' : [[0x590,0x5ff],[0xfb1d,0xfb4f],[0x5b0,0x5ff]] # TODO: Q/A
}
def count_lang_chars(string, lang):
    """
    Count num of chars in string that are within one of the character ranges for <lang>
    TODO: If we want sentences containing markup (html/wiki) - the ratio should be more tolerant
    """
    char_codes = np.array([ord(char) for char in string])
    char_ranges = lang_char_ranges[lang]    
    count = 0
    for char_range in char_ranges:
        count += np.sum((char_codes >= char_range[0]) & (char_codes <= char_range[1]))
    return count


def filter_sentences_df(df, lang, thr_lang_chars = 0.70):
    """

    Parameters
    ----------
    df : pandas dataframe.
    thr_lang_chars - at least 0.7 of the sentences chars should belong to the lang char ranges
    Returns
    -------
    He_filtered_df : filter the df to sentences with 4-30 words, that written only in Hebrew.

    """
    # filter the length of the sentences (between 4-45 words)
    df['word_count'] = df['HE_sentences'].apply(lambda x: count_words(x))
    len_filtered_df = df.query('word_count > 3 and word_count < 45').copy()

    # filter the df to only hebrew sentences
    print(len(len_filtered_df))
    len_filtered_df['lang_cnt'] = len_filtered_df['HE_sentences'].apply(lambda x: count_lang_chars(x,lang))
    He_filtered_df = len_filtered_df[len_filtered_df.lang_cnt > thr_lang_chars * len_filtered_df['HE_sentences'].str.len()]
   
    #len_filtered_df['sen_lang'] = len_filtered_df['HE_sentences'].apply(lambda x: detect_lang(x))
    #He_filtered_df = len_filtered_df.query(f"sen_lang == '{lang}'")

    He_filtered_df = He_filtered_df[['title', 'HE_sentences']]
    return He_filtered_df


def count_words(cell_content):
    words = cell_content.split()
    return len(words)


def detect_lang(cell_content):
    
    #return detect(cell_content)
    
    return 'unknown'


def split_df(df, he_folder_path=None):
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
    copied_df = copied_df.drop(columns=['title'], errors='ignore')

    for number in range(int(len(df) // 30000 + 1)):  # the max size for google translate
        if len(df) > 30000:
            part_df = copied_df.head(30000)
            copied_df = copied_df.drop(copied_df.index[:30000])
        else:
            part_df = copied_df

        if he_folder_path:
            part_df.to_excel(f'{he_folder_path}/{number}_part_HE.xlsx', index=False)
        else:
            part_df.to_excel(f'he_tr_excel/{number}_part_HE.xlsx', index=False)


def concat_dir_excels(folder_path):
    concat_df = pd.DataFrame()

    for file_path in Path(folder_path).glob("*.xlsx"):
        file_df = pd.read_excel(file_path)
        concat_df = pd.concat([concat_df, file_df], ignore_index=True)

    return concat_df


def create_concatenated_translated_df(he_folder_path='tr_data/he_tr_excel', en_folder_path='tr_data/en_tr_excel'):
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
    df = concat_dir_excels(he_folder_path)

    en_df = concat_dir_excels(en_folder_path)
    en_df = en_df.rename(columns={'HE_sentences': 'EN_sentences'})

    df['EN_sentences'] = en_df['EN_sentences']
    df = df.drop_duplicates(
        subset=['HE_sentences'])  # I translated the duplicates as well so I cant drop them in the beggining

    df = df.reset_index(drop=True)

    return df


if __name__ == '__main__':
    from dataset.gtranslate_selenium import google_translate_folder_of_excels

    # the 5 steps of translating a df

    df = pd.read_parquet('path/to/df')  # step 1: read the df
    split_df(df)  # step 2: split the df for Google translate

    he_folder_path = 'he_tr_excel'  # splited_df saves the files in this folder
    google_translate_folder_of_excels('he_tr_excel')  # step 3: translate the splited files

    # step 4: move the translated files from the downloads to new folder (or keep the download folder clean from irelevent excel files).

    en_folder_path = 'en_folder_path(change to your own)'
    create_concatenated_translated_df(he_folder_path, en_folder_path)  # step 5: concat everithing together

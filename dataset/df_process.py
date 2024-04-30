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


def count_words(cell_content):
    words = cell_content.split()
    return len(words)

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
    df = df.dropna(subset=['HE_sentences'])
    df['word_count'] = df.HE_sentences.apply(lambda x: count_words(x))
    len_filtered_df = df.query('word_count > 3 and word_count < 45').copy()

    # filter the df to only lang sentences
    print(len(len_filtered_df))
    len_filtered_df['lang_cnt'] = len_filtered_df['HE_sentences'].apply(lambda x: count_lang_chars(x,lang))
    He_filtered_df = len_filtered_df[len_filtered_df.lang_cnt > thr_lang_chars * len_filtered_df['HE_sentences'].str.len()]
    print(len(He_filtered_df))
    #len_filtered_df['sen_lang'] = len_filtered_df['HE_sentences'].apply(lambda x: detect_lang(x))
    #He_filtered_df = len_filtered_df.query(f"sen_lang == '{lang}'")

    He_filtered_df = He_filtered_df[['title', 'HE_sentences']]
    return He_filtered_df




def detect_lang(cell_content):
    
    #return detect(cell_content)
    
    return 'unknown'


def split_df(df, folder_path=None):
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

        if folder_path:
            part_df.to_excel(f'{folder_path}/{number}_part_HE.xlsx', engine='xlsxwriter', index=False)
        else:
            part_df.to_excel(f'split_excel_for_gtranslate/{number}_part_HE.xlsx', engine='xlsxwriter', index=False)




def translate_df(df_sents, dataset_prefix, src_lang, dst_lang):
    """
    Call GTranslate

    Parameters
    ----------
    df_sents : TYPE
        DESCRIPTION.
    dataset_prefix : TYPE
        DESCRIPTION.
    src_lang : TYPE
        DESCRIPTION.
    dst_lang : TYPE
        DESCRIPTION.

    Returns
    -------
    split_folder_path : TYPE
        DESCRIPTION.

    """
    from dataset.gtranslate_selenium import google_translate_folder_of_excels
    split_folder_path = Path(Path.cwd() / f'./{dataset_prefix}_split_excel_for_gtranslate/{src_lang}').resolve()  # splited_df saves the files in this folder in excel files
    split_folder_path.mkdir(parents=True, exist_ok=True)
    split_df(df_sents, split_folder_path)  # step 2: split the df for Google translate

    
    google_translate_folder_of_excels(split_folder_path, dst_lang)  # step 3: translate the splited files
    return split_folder_path

######## Concats resulting translated .xlsx (after copied from Downloads folder) #######
def concat_dir_excels(folder_path, common_files):
    concat_df = pd.DataFrame()

    for file_name in common_files:
        file_path = Path(folder_path) / file_name
        file_df = pd.read_excel(file_path)
        concat_df = pd.concat([concat_df, file_df], ignore_index=True)

    return concat_df

def create_concatenated_translated_df(src_folder_path='tr_data/he_tr_excel', dst_folder_path='tr_data/en_tr_excel', src_lang, dst_lang):
    """
    Parameters
    ----------
    src_folder_path : path to the folder that contains the split src_lang (Hebrew) sentences df
    en_folder : path to the folder that contains the split translated dst_lang (English) sentences df
    Returns
    -------
    concatenated df that contains the data of all of the files from both folders, drop the duplicates,
    and has 2 columns: HE_sentences, EN_sentences
    """
    src_files = set([file.name for file in Path(src_folder_path).glob("*.xlsx")])
    dst_files = set([file.name for file in Path(dst_folder_path).glob("*.xlsx")])

    common_files = src_files.intersection(dst_files)

    src_df = concat_dir_excels(src_folder_path, common_files)

    dst_df = concat_dir_excels(dst_folder_path, common_files)
    dst_df.columns = [f'EN_sentences']
    
    df = pd.concat([src_df.reset_index(drop=True), dst_df.reset_index(drop=True)], axis=1)
    df = df.drop_duplicates(subset=['HE_sentences'])

    return df

def create_translation_pairs_df(split_folder_path, dataset_prefix, src_lang, dst_lang):    
    # **** step 4: Manually move the translated files from the downloads to new folder (or keep the download folder clean from irelevent excel files).
    
    dst_folder_path = Path(f'{dataset_prefix}_split_excel_for_gtranslate/{dst_lang}')
    df_dataset = create_concatenated_translated_df(split_folder_path, dst_folder_path, src_lang, dst_lang)  # step 5: concat everithing together
    df_dataset.to_parquet(f'{dataset_prefix}_sentences_dataset_{src_lang}_translated_to_{dst_lang}.parquet')
    return df_dataset

if __name__ == '__main__':
    
    # the 5 steps of translating a df
    src_lang = 'fa'
    dst_lang = 'en'
    path_df_sentences = f'relevant_categories_sentences/{src_lang}/relevant_categories_sentences_{src_lang}.parquet'
    df_sents = pd.read_parquet(path_df_sentences)  # step 1: read the sentences df (after categories-relevant-pages + dump-parsing + sentence splitting and filtering)
    dataset_prefix='wikipedia'
    split_folder_path = translate_df(df_sents, dataset_prefix=dataset_prefix, src_lang=src_lang, dst_lang=dst_lang)
    # **** step 4: Manually move the translated files from the downloads to new folder (or keep the download folder clean from irelevent excel files).
    create_translation_pairs_df(split_folder_path, dataset_prefix=dataset_prefix, src_lang, dst_lang)
    
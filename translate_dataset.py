import string
from pathlib import Path
import numpy as np
import pandas as pd
import unittest
from loguru import logger


def translate_testset():
     from nlp_utils.translate.translate_ds import TranslateUtil
     # Create translation - Start with english test execution --> df_gt_prepro.parquet eng is saved
     # Important: Every change in fix_labels / fix types ... --> need to retranslate
     folder_path = './temp'
     target_lang = 'he'
     def read_df_first_letter_upper_case(path):
          df = pd.read_parquet(path)
          df['original_name'] = df['original_name'].apply(lambda x: string.capwords(x))
          df['matched_name'] = df['matched_name'].apply(lambda x: string.capwords(x))
          return df
     file_name = 'df_gt_prepro.parquet'
     trutil = TranslateUtil('./temp', [file_name], 
                            tr_cols = ['original_name','matched_name'], 
                            copy_cols = ['type','exception','match'], 
                            copy_rename_cols =['original_name','matched_name'], 
                            read_df_func = read_df_first_letter_upper_case)
     # 1) Save .xlsx files with eng, ready to upload
     trutil.save_eng_xls()
     # 2) Manually upload to Google Translate , the .xlsx files from eng_for_tr folder created in prev step --> save translations to ar
     # 3) create final files with column names in eng + copied columns from eng .xlsx
     ar_path = Path(folder_path) / target_lang
     trutil.process_tr_files(ar_path)
     df_gt_prepro_ar = pd.read_csv(Path(ar_path) / (Path(file_name).stem +'.csv') ,sep='\t',encoding='utf-8')
     # Filter out rows that are in english - not translated 
     RE_ENG_CHARS = r"[a-zA-Z._-]{3,}"
     df_gt_prepro_ar = df_gt_prepro_ar[~df_gt_prepro_ar.original_name.str.contains(RE_ENG_CHARS, regex = True) & ~df_gt_prepro_ar.matched_name.str.contains(RE_ENG_CHARS, regex = True)]
     # Filter out rows that orig name == match name (They weren't eq in eng, but after translation they are eq). 
     # There is neq filter at each run, but also here - allows us to explore the translated pairs 
     df_gt_prepro_ar = df_gt_prepro_ar[df_gt_prepro_ar.original_name.str.trim().str.lower() != df_gt_prepro_ar.matched_name.trim().str.str.lower()]
          
     # Save as .xlsx in the test folder - ready to commit
     df_gt_prepro_ar.to_excel( Path(GT_PATH) / target_lang / (Path(file_name).stem + '.xlsx') )

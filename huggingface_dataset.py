#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 20:43:31 2023

@author: aloncohen
"""

import pandas as pd
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset, Features, Value
from sklearn.model_selection import train_test_split
import os

def concat_translated_df(HE_df_path, EN_df_path):
    
    HE_df = pd.read_excel(HE_df_path)
    
    EN_df = pd.read_excel(EN_df_path)
    EN_df.rename(columns={'HE_sentences':'EN_sentences'}, inplace=True)
    
    concat_df = pd.concat([HE_df, EN_df], axis=1)
    concat_df = concat_df.drop(['Unnamed: 0'], axis=1)
    
    return concat_df

def df_to_dataset (df, train_split=0.8, test_split=0.1, eval_split=0.1):
   shuffled_df = df.sample(frac=1).reset_index(drop=True)
   
   train_size = int(len(shuffled_df) * train_split)
   eval_size = int(len(shuffled_df) * test_split)
   test_size = int(len(shuffled_df) * eval_split)
   
   train_data = shuffled_df[:train_size]
   val_data = shuffled_df[train_size:train_size+eval_size]
   test_data = shuffled_df[train_size+eval_size:]
   
   features = Features(
    {        
        "HE_sentences": Value("string"),
        "EN_sentences": Value("string")        
    })
   
   # Create a dictionary of datasets
   dataset_dict = {
        'train': Dataset.from_pandas(train_data, features=features),
        'validation': Dataset.from_pandas(val_data, features=features),
        'test': Dataset.from_pandas(test_data, features=features),
    }

   return DatasetDict(dataset_dict)

if __name__ == '__main__':
    
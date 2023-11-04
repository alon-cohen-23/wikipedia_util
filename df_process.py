# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 23:28:54 2023

@author: family123
"""

import pandas as pd 
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import requests

def filter_sentences_df (df):
   # filter the length of the sentences (between 4-30 words)
   df['word_count'] = df['HE_sentences'].apply(lambda x: count_words(x))
   len_filtered_df = df.query('word_count > 3 and word_count < 31').copy()
   
   #filter the df to only hebrew sentences
   len_filtered_df['sen_lang'] = len_filtered_df['HE_sentences'].apply(lambda x: detect_lang(x))
   He_filtered_df = len_filtered_df.query("sen_lang == 'he'")
   
   He_filtered_df = He_filtered_df['HE_sentences']
   return He_filtered_df
    
def count_words(cell_content):
    words = cell_content.split()
    return len(words)    

def detect_lang (cell_content):
    try:
        return detect(cell_content)
    except LangDetectException:
        return 'unknown'

def split_df (df):
    copied_df = df.copy() 
   
    for number in range (int(len(df)/30000+1)): # the max size for google translate 
        if len(df)>30000:
            part_df = copied_df.head(30000)
            copied_df = copied_df.drop(copied_df.index[:30000])
        else:
            part_df = copied_df
            
        part_df.to_excel(f'he_tr_excel/{number}_part_HE.xlsx')    
        
def upload_df_online(df_html_path):
        # URL of the website's file upload endpoint
    upload_url = "https://AlonWikiDf23/"
    
    # Create a dictionary to specify file details (name and content)
    files = {
        'file': (df_html_path, open(df_html_path, 'rb'), 'text/html')
    }
    
    # Send the POST request with the file and data
    response = requests.post(upload_url,  files=files)
    
    # Check the response
    if response.status_code == 200:
        print("File uploaded successfully.")
    else:
        print(f"Failed to upload the file. Status code: {response.status_code}")
   
    
    
    
    
    
                
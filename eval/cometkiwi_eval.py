# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:59:46 2023

@author: DEKELCO
"""

# !pip install unbabel-comet

import time
from pathlib import Path
import numpy as np
import pandas as pd
from comet import download_model, load_from_checkpoint
from huggingface_hub import login
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = "cuda"

def hf_login(access_token):
    """
    Logs into HF, so we can download comet models - which are not public without login (anon) 
    
    Parameters
    ----------
    access_token : Huggingface access token - see HF WebSite | Profile (upper left) | settings | Access Tokens    
    """
    login(token=access_token)    
   
def get_col_value(row,col_path):
    """
    Helper to get value from df col which may be a simple column or a column containing a dict

    Parameters
    ----------
    row : df row 
    col_path : '<col name string>' or ['col name','dict key inside col']
    
    Returns
    -------
    val 

    """
    if isinstance(col_path, str):
        col_path = [col_path]
    elif not isinstance(col_path, list):
        raise Exception("col_path must be either a string ('HE_sentences') or a list of strings ['translation']['he']")
    val = row
    for cur_seg in col_path:
        val = val.get(cur_seg)
    return val
           
def calc_comet_score(comet_model, df_samp, col_src, col_dst, col_ref=None):
    """
    Parameters
    ----------
    df_samp : df containing source and target translation columns
    col_src : name of source lang col
    col_dst : name of target lang col 
    col_ref : name of reference (GT) translation text to compare against target col. If None (default) - score without reference
    Returns
    -------
    None.

    """
    data = []
    for index, row in df_samp.iterrows():  
        src_val = get_col_value(row, col_src)
        dst_val = get_col_value(row, col_dst)
        
        data.append({  
            'src': src_val,  
            'mt': dst_val
        })  
      
    model_output = comet_model.predict(data, batch_size=64, gpus=1)
    df_samp['comet_score'] = model_output[0]    
    return df_samp


def predict(tokenizer, model ,df_samp, col_src, dst_lang="eng_Latn", batch_size = 500):
    src_texts = []
    for index, row in df_samp.iterrows():  
        src_val = get_col_value(row, col_src)
        src_texts.append(src_val)
 
    start = time.perf_counter()
    # Split the texts into batches  
    translated_texts = []
    batches = [src_texts[i:i + batch_size] for i in range(0, len(src_texts), batch_size)]     
    for batch in batches:  
        inputs = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True ).to(device)    
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[dst_lang])
        translated_texts += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[:]
    
    tot_time = time.perf_counter() - start
    print(f'predict time: {tot_time}')

    return translated_texts

def main(model_name_or_path, test_df_path = './data/validation.parquet', max_samples = 4000):     
    # make predictions (translations)
    src_lang= "pes_Arab" # "arb_Arab" # "heb_Hebr"
    col_src= ['translation','he'] # 'HE_sentences'
    col_dst='pred'
    dst_lang="eng_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, src_lang=src_lang)
    model = model.to_bettertransformer()
    
    # Read samples, in HF read df row['translation']['he'] and row['translation']['en']
    testset_path = Path(test_df_path)
    df_samp = pd.read_parquet(testset_path)
    
    df_samp = df_samp[:max_samples]
    translated_texts = predict(tokenizer, model ,df_samp, col_src=col_src, dst_lang=dst_lang)
    df_samp['pred'] = translated_texts
    
    
    # login to huggingface in order to download comet model
    # hf_login(access_token) # get token from https://colab.research.google.com/drive/1ULigrTC9ppf2pc0a1-D6cewx0aOGGTkX#scrollTo=hhxqbEN_9FkW
   
    comet_model_name = 'Unbabel/wmt22-cometkiwi-da' #'Unbabel/wmt23-cometkiwi-da-xl'
    comet_model_path = download_model(comet_model_name)
    comet_model = load_from_checkpoint(comet_model_path)
    # TODO:HIGH:Restore: col_dst='pred' or col_dst=['translation']['en']
    df_samp = calc_comet_score(comet_model=comet_model, df_samp=df_samp, col_src=col_src, col_dst=col_dst)
    Path('./temp').mkdir(exist_ok=True)
    df_samp.to_parquet('./temp/test_commet.parquet')
    df_samp[df_samp.comet_score < 0.50][:200].to_html('./temp/bad_lt_0_5.html')
    df_samp[(df_samp.comet_score > 0.55) & (df_samp.comet_score < 0.7)][:200].to_html('./temp/good_gt_0_55_lt_0_7.html')
    df_samp[df_samp.comet_score > 0.7][:200].to_html('./temp/good_gt_0_7.html')
    print('Bad translation ratio',df_samp[df_samp.comet_score < 0.50].shape[0] / df_samp.shape[0])
    descrb = df_samp[['comet_score']].describe(percentiles=np.arange(0, 1, 0.1))
    print(descrb)
    
    descrb.to_html('./temp/commet_score_quantiles.html')

    # Open the file in write mode
    with open("./temp/model_training_metadata.txt", "w") as file:
        # Write the string into the file
        file.write(f'model_name_or_path={model_name_or_path}')
    return df_samp
    
if __name__ == "__main__":    
    model_name_or_path_3_3 = 'output_models/nllb-200-3.3B_arb_eng_v2/checkpoint-419710/'
    model_name_or_path_600 = 'output_models/nllb-200-distilled-600M_arb_eng_telegram_v2/checkpoint-209856'
    model_name_or_path_600_he_en = './nllb-200-distilled-600M_heb_eng/checkpoint-172058'
    model_name_or_path_600_fa_en_wikipedia = './output_models/nllb-200-distilled-600M_pes_eng/checkpoint-124185'
    df_samp = main(model_name_or_path = model_name_or_path_600_fa_en_wikipedia) # 'output_models/nllb-200-distilled-600M_heb_eng/checkpoint-172058/'
                   
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:59:46 2023

@author: DEKELCO
"""

# !pip install unbabel-comet

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
        data.append({  
            'src': row['translation'][col_src],  
            'mt': row[col_dst]  # TODO:HIGH:Restore: row[col_dst] row['translation']['en']  
        })  
      
    model_output = comet_model.predict(data, batch_size=64, gpus=1)
    df_samp['comet_score'] = model_output[0]    
    return df_samp


def predict(tokenizer, model ,df_samp, col_src='he', dst_lang="eng_Latn", batch_size = 500):
    src_texts = []
    for index, row in df_samp.iterrows():  
        src_texts.append(row['translation'][col_src])
  
    # Split the texts into batches  
    translated_texts = []
    batches = [src_texts[i:i + batch_size] for i in range(0, len(src_texts), batch_size)]     
    for batch in batches:  
        inputs = tokenizer.batch_encode_plus(batch, return_tensors="pt", padding=True ).to(device)    
        translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[dst_lang])
        translated_texts += tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[:]
    return translated_texts

def main(model_name_or_path, max_samples = 4000):     
    # make predictions (translations)
    src_lang="heb_Hebr"
    col_src='he'
    dst_lang="eng_Latn"
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, src_lang=src_lang)
    model = model.to_bettertransformer()
    
    # Read samples, in HF read df row['translation']['he'] and row['translation']['en']
    testset_path = Path('./data/validation.parquet')
    df_samp = pd.read_parquet(testset_path)
    
    df_samp = df_samp[:max_samples]
    translated_texts = predict(tokenizer, model ,df_samp, col_src=col_src, dst_lang=dst_lang)
    df_samp['pred'] = translated_texts
    
    
    # login to huggingface in order to download comet model
    # hf_login(access_token) # get token from https://colab.research.google.com/drive/1ULigrTC9ppf2pc0a1-D6cewx0aOGGTkX#scrollTo=hhxqbEN_9FkW
    
    comet_model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    comet_model = load_from_checkpoint(comet_model_path)
    df_samp = calc_comet_score(comet_model=comet_model, df_samp=df_samp, col_src=col_src, col_dst='pred')
    Path('./temp').mkdir(exist_ok=True)
    df_samp.to_parquet('./temp/test_commet.parquet')
    df_samp[df_samp.comet_score < 0.50][:200].to_html('./temp/bad_lt_0_5.html')
    df_samp[df_samp.comet_score > 0.7][:200].to_html('./temp/good_gt_0_7.html')
    print('Bad translation ratio',df_samp[df_samp.comet_score < 0.50].shape[0] / df_samp.shape[0])
    descrb = df_samp[['comet_score']].describe(percentiles=np.arange(0, 1, 0.1))
    print(descrb)
    
    descrb.to_html('./temp/commet_score_quantiles.html')
    return df_samp
    
if __name__ == "__main__":    
    df_samp = main(model_name_or_path = 'output_models/nllb-200-distilled-600M_heb_eng/checkpoint-172058/')
    # "/data2/NLP/translation/nllb/output_models/nllb-200-distilled-1.3B_heb_eng_wiki_40000/checkpoint-80448" # "facebook/nllb-200-distilled-1.3B"
    

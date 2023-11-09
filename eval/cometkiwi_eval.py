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
    print('Bad translation ratio',df_samp[df_samp.comet_score < 0.50].shape[0] / df_samp.shape[0])
    print(df_samp[['comet_score']].describe(percentiles=np.arange(0, 1, 0.1)))
    return df_samp
    
if __name__ == "__main__":
    #hf_login(access_token) # get token from https://colab.research.google.com/drive/1ULigrTC9ppf2pc0a1-D6cewx0aOGGTkX#scrollTo=hhxqbEN_9FkW
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    results_path = Path('./outputs/predictions.parquet')
    df_samp = pd.read_parquet(results_path)
    comet_model = load_from_checkpoint(model_path)
    df_samp = calc_comet_score(comet_model=comet_model, df_samp=df_samp, col_src='he', col_dst='pred')
    df_samp[df_samp.comet_score < 0.50]

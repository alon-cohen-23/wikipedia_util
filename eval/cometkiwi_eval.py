# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:59:46 2023

@author: DEKELCO
"""

# !pip install unbabel-comet

from pathlib import Path
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

def calc_comet_score(comet_model, df, col_src, col_dst, col_ref=None):
    """
    

    Parameters
    ----------
    df : df containing source and target translation columns
    col_src : name of source lang col
    col_dst : name of target lang col 
    col_ref : name of reference (GT) translation text to compare against target col. If None (default) - score without reference
    Returns
    -------
    None.

    """
    
    data = df_samp.rename(columns={"he": "src", "en": "mt"}).to_dict(orient="records")  
    model_output = comet_model.predict(data, batch_size=64, gpus=1)


if __name__ == "__main__":
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    results_path = Path('./outputs/predictions.parquet')
    df = pd.read_parquet(results_path)
    model = load_from_checkpoint(model_path)
    calc_comet_score(comet_model=model, df, col_src='HE_sentences', col_dst='EN_sentences')
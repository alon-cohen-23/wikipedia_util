from pathlib import Path
import pandas as pd
from eng_utils.common.common_util import read_df_folder
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
from tqdm import tqdm

tqdm.pandas()



CSV_ARTICLES_FOLDER_PATH = r"D:\NLP\Translation\wikipedia_util\data\articles\papers_graph_1000"


if __name__ == "__main__":
    df = read_df_folder(CSV_ARTICLES_FOLDER_PATH, glob_pattern = "*.csv", read_func=pd.read_csv)    
    df = df.dropna(subset=['abstract'])
    #df.iloc[0].abstract
    
    sents = df.progress_apply(lambda x: sum([sent_tokenize(y) for y in LineTokenizer(blanklines='discard').tokenize(x.abstract)], []), axis=1)
    sents = sents.explode()
    df_sents = sents.to_frame().rename(columns={'0' : 'sent'}).reset_index(drop=True)

# cd D:\NLP\Translation\wikipedia_util

# TODO: Filter sentences (len)
# TODO: Fix col names HE_sentences - fa EN_sentences - en
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
    df_sents = sents.to_frame().rename(columns={0 : 'HE_sentences'}).reset_index(drop=True)

    from dataset.df_process import translate_df,create_translation_pairs_df
    src_lang = 'en'
    dst_lang = 'fa'        
    dataset_prefix='articles'
    split_folder_path = translate_df(df_sents, dataset_prefix=dataset_prefix, src_lang=src_lang, dst_lang=dst_lang)
    # **** step 4: Manually move the translated files from the downloads to new folder (or keep the download folder clean from irelevent excel files).
    #split_folder_path = './articles_split_excel_for_gtranslate/en'
    df_dataset = create_translation_pairs_df(split_folder_path, dataset_prefix=dataset_prefix, src_lang=src_lang, dst_lang=dst_lang)
    df_dataset.columns = df_dataset.columns[::-1] # Since we translated en --> fa and the code is used to translating to en - reverse src and dst columns    
    df_dataset.to_parquet(f'{dataset_prefix}_sentences_dataset_{src_lang}_translated_to_{dst_lang}.parquet')
    #print(df_dataset.iloc[1320].EN_sentences)
    #print(df_dataset.iloc[1320].HE_sentences)

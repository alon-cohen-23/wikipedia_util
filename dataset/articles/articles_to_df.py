# cd D:\NLP\Translation\wikipedia_util\dataset\articles

# TODO: Filter sentences (len)
# TODO: Fix col names HE_sentences - fa EN_sentences - en
from pathlib import Path
import pandas as pd
from eng_utils.common.common_util import read_df_folder
from nltk.tokenize import sent_tokenize
from nltk.tokenize import LineTokenizer
from tqdm import tqdm

tqdm.pandas()


# "G:\My Drive\Syncy\Projects\EntsRels\Translation_NLLB\scientific_illaya_articles\papers_graph_1000.zip"



if __name__ == "__main__":
    dataset_prefix='articles'
    suffix = '1000' # 1000 abstracts
    src_lang = 'en'
    dst_lang = 'he' # 'fa'        
    CSV_ARTICLES_FOLDER_PATH = rf"D:\NLP\Translation\wikipedia_util\dataset\articles\data_raw\{suffix}"    
    df = read_df_folder(CSV_ARTICLES_FOLDER_PATH, glob_pattern = "*.csv", read_func=pd.read_csv)    
    df = df.dropna(subset=['abstract'])
    #df.iloc[0].abstract
    
    sents = df.progress_apply(lambda x: sum([sent_tokenize(y) for y in LineTokenizer(blanklines='discard').tokenize(x.abstract)], []), axis=1)
    sents = sents.explode()
    df_sents = sents.to_frame().rename(columns={0 : 'HE_sentences'}).reset_index(drop=True)

    from dataset.df_process import translate_df,create_translation_pairs_df
    
    split_folder_path = translate_df(df_sents, dataset_prefix=dataset_prefix, src_lang=src_lang, dst_lang=dst_lang)
    # **** step 4: Manually move the translated files from the downloads to new folder (or keep the download folder clean from irelevent excel files).
    # Copy to (en-->he): D:\NLP\Translation\wikipedia_util\dataset\articles\split_excel_for_gtranslate\articles\he
    #split_folder_path = './articles_split_excel_for_gtranslate/en'
    df_dataset, p_ds = create_translation_pairs_df(split_folder_path, dataset_prefix=dataset_prefix,suffix=suffix, src_lang=src_lang, dst_lang=dst_lang)
    df_dataset.columns = df_dataset.columns[::-1] # Since we translated en --> fa and the code is used to translating to en - reverse src and dst columns    
    df_dataset.to_parquet(p_ds)
    #print(df_dataset.iloc[1320].EN_sentences)
    #print(df_dataset.iloc[1320].HE_sentences)

from pathlib import Path
import pandas as pd
from eng_utils.common.common_util import read_df_folder


if __name__ == "__main__":
    data_folder = Path(r"D:\NLP\Translation\wikipedia_util\data\raw")
    df = read_df_folder(data_folder)
    df.info()
    df= df.rename(columns={'text.en' : 'EN_sentences', 'text.ar' : 'HE_sentences'})
    df = df[['EN_sentences', 'HE_sentences']]
    df.to_parquet(data_folder / 'telegram_sents_ar_en.parquet')
    df.iloc[10001].EN_sentences
    df.iloc[10001].HE_sentences
    
    df_t =  pd.read_parquet(r"D:\NLP\Translation\wikipedia_util\data\raw\twitter\df_day_1_11_2023.parquet")
    df_t.info()
    df_t.iloc[1500]

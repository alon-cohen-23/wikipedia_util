import pandas as pd
from pathlib import Path

from dataset.df_process import filter_sentences_df, split_df, create_concatenated_translated_df
from dataset.gtranslate_selenium import google_translate_folder_of_excels
from dataset.text_files.read_and_translate_text_files import read_all_files


def preprocess_df(df):
    max_input_length = 200
    max_target_length = 200

    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])

    df = df[df.HE_sentences.str.len() <= max_input_length]
    df = df[df.EN_sentences.str.len() <= max_target_length]
    df['translation'] = df.apply(lambda row: {'en': row['EN_sentences'], 'he': row['HE_sentences']}, axis=1)

    # Drop the original 'EN_sentences' and 'HE_sentences' columns
    return df.drop(columns=['EN_sentences', 'HE_sentences'])


def full_flow(input_path, input_name, output_root, steps):
    """
    Run a full flow for preparing a dataset. The input parameters steps specifies what steps should run.
    If a step exists in the list then it will be run. Otherwise, it will be either be skipped or read from disk.
    :param input_path: path to input
    :param input_name: name of input (e.g. wiki, inss). Will affect the directory naming.
    :param output_root: root directory for all the output files
    :param steps: list of steps (range 1->6).
    """
    output_dir = Path(output_root)
    output_named = output_dir / f"{input_name}"
    he_folder_path = output_named / "he_tr_excel"
    en_folder_path = output_named / "en_tr_excel"
    he_folder_path.mkdir(parents=True, exist_ok=True)
    en_folder_path.mkdir(parents=True, exist_ok=True)

    input_pages_file = output_named / "all_pages.parquet"
    output_translated_file = output_named / "translated_{input_name}.parquet"
    output_translated_file_final = output_named / "translated_{input_name}_final.parquet"

    # step 1: prepare the df
    if 1 in steps:
        # currently, assumes input is a collection of text file which possibly isn't true for wiki data collected
        # need to adapt the wikipedia code and call it here if input_name=="wiki"
        # this code assumes the structure of the
        df = read_all_files(input_path)
        df = filter_sentences_df(df)  # filter the df by calling the function from df_process.py
        df.to_parquet(input_pages_file)
    else:
        df = pd.read_parquet(input_pages_file)

    # step 2: split the df for Google translate
    if 2 in steps:
        split_df(df, he_folder_path)

    # step 3: translate the split files
    if 3 in steps:
        google_translate_folder_of_excels(he_folder_path)

    if 4 in steps:
        print(f"This step is manual: copy files from downloads into {en_folder_path}!")
    # step 4 (manual): move the translated files from the downloads to new folder
    # (or keep the download folder clean from irrelevant Excel files).

    # step 5: concat everything together
    if 5 in steps:
        en_he_df = create_concatenated_translated_df(he_folder_path, en_folder_path)
        en_he_df.to_parquet(output_translated_file)
    else:
        en_he_df = pd.read_parquet(output_translated_file)

    # step 6: organize the dataframe so it is ready for pipeline
    if 6 in steps:
        en_he_df = preprocess_df(en_he_df)
        en_he_df.to_parquet(output_translated_file_final)


if __name__ == '__main__':
    full_flow(input_path=r"/workspace/translation/inss/data", input_name="inss", output_root=r"/workspace/translation",
              steps=[1, 2, 3, 4, 5, 6])

import pandas as pd
from sentence_splitter import sentenceSplitterDicta


def remove_wrong_acronym(df):
    def is_wrong_acronym(x):
        splitted = x.split('"')
        return len(splitted[0]) == 1 and len(splitted[1]) > 1

    df = df[~df.index.map(is_wrong_acronym)]
    return df

def identify_base(df_orig):
    df = df_orig.copy()
    df.sort_values(by='base_acronym', ascending=False, key=lambda col: col.str.len(), inplace=True)
    for i in range(len(df.base_acronym) - 1):
        for j in range(i + 1, len(df.base_acronym)):
            if len(df.base_acronym[j]) <= 3: break
            if df.base_acronym[i].startswith(df.base_acronym[j]):
                df.loc[i, 'base_acronym'] = df.base_acronym[j]
    return df

def group_by_base(df):
    def acronyms_data_agg(s):
        if s.dtype == 'O':
            return list(s)
        else:
            return sum(s)

    df_count_base = df.groupby('base_acronym').agg(acronyms_data_agg)
    df_count_base['n_united'] = df_count_base.acronym.apply(len)
    return df_count_base

def create_acronym_list(acronym_src, output_file):
    if type(acronym_src)==str: # filename
        df_counts = pd.read_csv(acronym_src, encoding='utf-8', index_col = 'acronym')
    else: # df
        df_counts = acronym_src.copy()
    sentence_splitter = sentenceSplitterDicta()
    df_counts = remove_wrong_acronym(df_counts).reset_index()
    df_counts['base_acronym'] = df_counts.acronym.apply(lambda token:
                                                        sentence_splitter.get_base_word(sentence_splitter.split_sentence(token)[0]))
    df_counts = identify_base(df_counts)
    df_count_base = group_by_base(df_counts)
    df_count_base.to_csv(output_file, encoding='utf-8')

if __name__=='__main__':
    create_acronym_list(acronym_src='data//acronyms_in_resources_small.csv',
                        output_file='data//outputs//acronyms_list_sample.csv')
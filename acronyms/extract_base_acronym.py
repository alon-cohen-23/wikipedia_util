import pandas as pd

def remove_wrong_acronym(df):
    def is_wrong_acronym(x):
        splitted = x.split('"')
        return len(splitted[0]) == 1 and len(splitted[1]) > 1

    df = df[~df.index.map(is_wrong_acronym)]
    return df

def identify_base(df_orig):
    df = df_orig.copy().reset_index()
    df['base'] = df.acronym
    df.sort_values(by='base', ascending=False, key=lambda col: col.str.len(), inplace=True)
    for i in range(len(df.base) - 1):
        for j in range(i + 1, len(df.base)):
            if len(df.base[j]) <= 3: break
            if df.base[j] in df.base[i]:
                # print(f"{i}, {j}, Old base: {df.base[i]}, New base: {df.base[j]}")
                df.loc[i, 'base'] = df.base[j]
    # print(df[df.base!=df.acronym])
    return df

def group_by_base(df):
    def acronyms_data_agg(s):
        if s.dtype == 'O':
            return list(s)
        else:
            return sum(s)

    df_count_base = df.groupby('base').agg(acronyms_data_agg)
    return df_count_base

if __name__=='__main__':
    df_counts = pd.read_csv('data//acronyms_in_resources_small.csv', index_col = 'acronym')
    df_counts = remove_wrong_acronym(df_counts)

    df_counts = identify_base(df_counts)

    df_count_base = group_by_base(df_counts)
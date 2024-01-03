from extract_base_acronym import create_acronym_list
import pandas as pd
import json

def json_acronyms_to_df_counts(json_file, n_min = 3):
    with open(json_file, encoding='utf-8') as f:
        acronyms_json = json.load(f)

    l = []

    for k, val in acronyms_json['acronyms'].items():
        d = {'acronym': val['name']}
        for source in val['sources']:
            d[f"{source['sname']}_counts"] = int(source['count'])
        l.append(d)
    df = pd.DataFrame.from_records(l, index='acronym').fillna(0)
    df['total_counts'] = df.sum(axis=1)
    # consider only those which appears equal or more than n_min times
    df = df[df.total_counts>n_min]
    print(df.head())
    return df

if __name__=='__main__':
    json_file = '''data\\acronyms_external_data.json'''
    df_counts = json_acronyms_to_df_counts(json_file)
    create_acronym_list(df_counts, 'data\\outputs\\acronyms_base_external.csv')
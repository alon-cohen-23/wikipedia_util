from itertools import product
from collections import defaultdict, Counter
from tqdm import tqdm
import pandas as pd
import pickle

from acronyms_utils import *
from sentence_splitter import sentenceSplitterDicta #split_sentence, detokenize_sentence, SPLIT_MODE

def get_first_n_grams(tokens, n, sentence_splitter):
    n_words_found = 0
    word_tokens = []
    for token in tokens:
        base_word = sentence_splitter.get_base_word(token)
        word = get_word_by_regex(base_word)
        if word is not None:
            n_words_found += 1
            word_tokens.append(token)
        if n_words_found == n:
            return word_tokens
    return None


def get_n_grams_after(tokens, n, sentence_splitter):
    res = get_first_n_grams(tokens, n, sentence_splitter )
    # print(f'after: {res}')
    return res

def get_n_grams_before(tokens, n, sentence_splitter):
    reversed_tokens = tokens[::-1]
    reversed_words = get_first_n_grams(reversed_tokens, n, sentence_splitter)
    res = reversed_words[::-1] if reversed_words else None
    return res

def is_possible_opened_form(acronym_splitted, words):
    return all([words[i].startswith(acronym_splitted[i]) for i in range(len(acronym_splitted))])

def create_partitions(l, k=2):
    if len(l) == 0: return [l]
    if len(l) == 1: return [[l]]
    full_res = []
    for i in range(1, min(k, len(l)) + 1):
        pref = l[:i]
        partial_partitions = create_partitions(l[i:], k)
        for pp_index in range(len(partial_partitions)):
            partial_partitions[pp_index] = [pref, *partial_partitions[pp_index]]
        full_res += partial_partitions
    return full_res

def find_opened_for_this_acronym(tokens, i, sentence_splitter):
    acronym = tokens[i]
    acronym_with_possible_prefixes = identify_possible_word_parts(acronym, sentence_splitter,
                                                                  last_index_indicator='"',
                                                                  check_for_prefixes=False)
    for ac_part in acronym_with_possible_prefixes:
        acronym_clean = clean_acronym(ac_part)
        acronym_chars_partitions = create_partitions(acronym_clean)
        for acronym_partition in acronym_chars_partitions:
            acronym_len = len(acronym_partition)
            n_grams_to_scan = []
            if i < len(tokens) - 1:
                words_after = get_n_grams_after(tokens[i + 1:], acronym_len, sentence_splitter)
                if words_after:
                    n_grams_to_scan.append(words_after)
            if i > 0:
                words_before = get_n_grams_before(tokens[:i], acronym_len, sentence_splitter)
                if words_before:
                    n_grams_to_scan.append(words_before)
            for n_gram in n_grams_to_scan:
                words_parts = []
                for w in n_gram:
                    words_parts.append(list(sentence_splitter.identify_possible_word_parts(w,
                                                                                           sentence_splitter, check_for_prefixes=True)))
                for words_parts_combination in product(*words_parts):
                    # print(words_parts_combination)
                    opened_form = is_possible_opened_form(acronym_partition, words_parts_combination)
                    if opened_form:
                        return sentence_splitter.get_base_word(acronym), words_parts_combination
    return None, []


def create_local_acronym_info(document, sentence_splitter):
    res = []
    if document is None or '"' not in document: return res
    tokens = sentence_splitter.split_sentence(document)
    for i in range(len(tokens)):
        word = sentence_splitter.get_base_word(tokens[i])
        if is_acronym(word, search_type='match'):
            # TODO: add support for acronyms with prefixes and suffixes
            acronym, words_parts_combination = find_opened_for_this_acronym(tokens, i, sentence_splitter)
            if acronym:
                res.append((acronym, sentence_splitter.detokenize_sentence(words_parts_combination)))
    return res

def collect_opened_form(sources, res_filename):
    d_res = {}
    sentence_splitter = sentenceSplitterDicta()
    i = 0
    for filename, src in sources:
        df = pd.read_parquet(filename)
        for document in tqdm(df.HE_sentences.values):
            res = create_local_acronym_info(document, sentence_splitter)
            if res!=[]:
                for (acronym, opened_form) in res:
                    print(f'Found: {acronym}, {opened_form}, {src}')
                    if acronym not in d_res:
                        d_res[acronym] = {}
                    if opened_form not in d_res[acronym]:
                        d_res[acronym][opened_form] = Counter()
                    d_res[acronym][opened_form][src] +=1
        i+=1
        if i%1000==0:
            print(i)
            pickle.dump(d_res, open(res_file, 'wb'))

def results2df(res_filename, df_file):
    d_res = pickle.load(open(res_filename, 'rb'))
    l_res = []
    # print(d_res['א"א'])
    # d_res['א"א'][' איחוד אירופי']['TEMP'] = 1000
    for acronym in d_res:
        for opened_form in d_res[acronym]:
            for src in d_res[acronym][opened_form]:
                count = d_res[acronym][opened_form][src]
                l_res.append((acronym, opened_form, src, count))
    df = pd.DataFrame(l_res, columns = ['acronym', 'opened_form', 'src', 'count']) #.set_index('acronym')
    df1 = df.pivot(index=['acronym', 'opened_form'], columns='src', values='count').reset_index()
    df1.to_csv(df_file)

if __name__=='__main__':
    # document = 'האסטרטגיה הישראלית בזירה זו בשנים האחרונות זכתה לכותרת "המערכה שבין המלחמות" (מב"מ).'
    # # document = 'הרמטכ"ל (ראש המטה הכללי) הורה לעשות את זה'
    # # document = 'צבא ההגנה לישראל (צה"ל) ינצח'
    res_filename = 'data\\outputs\\local_acronyms.pickle'
    sources = [('''C:\\Users\\MICHALD2\\projects\\Translator\\Michal\\sentences\\inss_all_pages.parquet''', 'inss'),
               ('''C:\\Users\\MICHALD2\\projects\\Translator\\Michal\\sentences\\translated_df_relevant_cats.parquet''', 'wiki')]
    df_file = 'data\\outputs\\df_local_dict.csv'
    # collect_opened_form(sources, res_filename)
    print('finished collecting opened form')
    results2df(res_filename, df_file)
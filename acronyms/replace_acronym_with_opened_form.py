import pandas as pd
import regex
from acronyms_utils import *
from sentence_splitter import split_sentence, detokenize_sentence


TRIE_SUPPORT = False
if TRIE_SUPPORT:
    from trie import Trie

def create_acronym_container(acronyms):
    if TRIE_SUPPORT:
        trie = Trie()
        trie.insert_batch(acronyms)
        return trie
    else:
        acronyms_sorted_by_length = sorted(acronyms, key=lambda x: -len(x))
        return acronyms_sorted_by_length


def get_acronym_datastructures(filename):
    s = pd.read_csv(filename, index_col=0)
    d_meaning = {key: val for (key, val) in s.meaning.items() if len(key) >= 3}
    acronyms_container = create_acronym_container(d_meaning.keys())
    # acronyms_sorted_by_length = sorted(d_meaning.keys(), key=lambda x: -len(x))
    return d_meaning, acronyms_container


def search_acronym_container(word, acronyms_container):
    if TRIE_SUPPORT:
        return acronyms_container.search_prefix(word)
    else:
        for acronym in acronyms_container:
            if word.startswith(acronym):
                # print('***', acronym)
                return acronym
        return None



def search_sub_acronym(word, acronyms_container=None):
    for word_part in identify_possible_word_parts(word, '"'):
        if acronyms_container is not None:
            acronym = search_acronym_container(word_part, acronyms_container)
            if acronym:
                return acronym
    return None


def add_acronym_meaning(sentence, d_meaning, acronyms_sorted):
    modified_acronyms = set()

    splitted = split_sentence(sentence, mode='regex')
    for i in range(len(splitted)):
        word = splitted[i]

        pattern = ACRONYM_REGEX_PATTERN #'\p{Hebrew}+\"\p{Hebrew}+'
        if regex.match(pattern, word) is not None:
            acronym = search_sub_acronym(word, acronyms_sorted)
            if acronym:
                modified_acronyms.add(acronym)
                splitted[i] = f'{word} ({d_meaning[acronym]})'
            # else:
            #     print(f'{word} not found in dict')

    new_sentence = detokenize_sentence(splitted)
    return new_sentence, modified_acronyms
def test(d_meaning, acronyms_sorted):
    sentence = 'בא"מ מתכננים לעשות את זה'
    print(add_acronym_meaning(sentence, d_meaning, acronyms_sorted))

if __name__=='__main__':
    d_meaning, acronyms_container = get_acronym_datastructures(
        'C:\\Users\\MICHALD2\\projects\\Translator\\wikipedia_util\\acronyms\\data\\output_acronyms_from_wiktionary.csv')
    # test(d_meaning, acronyms_sorted)

    par_file = 'C:\\Users\\MICHALD2\\projects\\Translator\\Michal\\sentences\\all_pages.parquet'
    df_inss = pd.read_parquet(par_file, engine='pyarrow')
    # df_inss.head()

    counter = 0
    modified = set()
    for i, row in df_inss.iterrows():
        sentence = row['HE_sentences']
        new_sentence, is_modified = add_acronym_meaning(sentence, d_meaning, acronyms_container)
        # if is_modified:
        if len(is_modified) > 0:
            if len(is_modified.intersection(modified)) == 0:
                counter += 1
                print('------')
                print(sentence)
                print(new_sentence)
            modified.update(is_modified)
        if counter == 10: break
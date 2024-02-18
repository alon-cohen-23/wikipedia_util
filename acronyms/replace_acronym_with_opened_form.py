import pandas as pd
import regex
from acronyms_utils import *
# from sentence_splitter import split_sentence, detokenize_sentence, SPLIT_MODE
from sentence_splitter import sentenceSplitterDicta, sentenceDefaultSplitter

TRIE_SUPPORT = True
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
    return d_meaning, acronyms_container

def search_acronym_container(word, acronyms_container):
    if TRIE_SUPPORT:
        return acronyms_container.search_prefix(word)
    else:
        for acronym in acronyms_container:
            if word.startswith(acronym):
                return acronym
        return None

def search_sub_acronym(word, sentence_splitter, acronyms_container=None, check_for_prefixes=False):
    possible_word_parts = sentence_splitter.identify_possible_word_parts(word, '"', check_for_prefixes)
    for word_part in possible_word_parts:
        if acronyms_container is not None:
            acronym = search_acronym_container(word_part, acronyms_container)
            if acronym:
                return acronym
    return None


def add_acronym_meaning(sentence, d_meaning, acronyms_container, sentence_splitter):
    if is_acronym(sentence, search_type='search') is None:
        return sentence, []

    # check_for_prefixes = SPLIT_MODE!='dicta'
    check_for_prefixes = not isinstance(sentence_splitter, sentenceSplitterDicta)

    modified_acronyms = set()
    splitted = sentence_splitter.split_sentence(sentence)
    for i in range(len(splitted)):
        base_word = sentence_splitter.get_base_word(splitted[i])
        # if SPLIT_MODE=='dicta': # was splitted with dicta
        #     word = splitted[i][-1]
        # else:
        #     word = splitted[i]

        if is_acronym(base_word, search_type='match') is not None:
            acronym = search_sub_acronym(splitted[i], sentence_splitter, acronyms_container, check_for_prefixes)
            if acronym:
                modified_acronyms.add(acronym)
                new_word =f'{splitted[i][-1]} ({d_meaning[acronym]})'
                # if SPLIT_MODE=='dicta':  # was splitted with dicta
                if isinstance(sentence_splitter, sentenceSplitterDicta):
                    splitted[i][-1] = new_word
                else:
                    splitted[i] = new_word
            # else:
            #     print(f'{word} not found in dict')

    new_sentence = sentence_splitter.detokenize_sentence(splitted)
    return new_sentence, modified_acronyms

if __name__=='__main__':
    d_meaning, acronyms_container = get_acronym_datastructures(
        'C:\\Users\\MICHALD2\\projects\\Translator\\wikipedia_util\\acronyms\\data\\outputs\\output_acronyms_from_wiktionary.csv')


    sentence_splitter =  sentenceSplitterDicta() #sentenceDefaultSplitter()

    sentence = '''התערבות בפוליטיקה הפנימית בישראל באמצעות הפעלת לחץ ישיר על רע"ם לפרוש מהממשלה.'''
    new_sentence, modified = add_acronym_meaning(sentence, d_meaning, acronyms_container, sentence_splitter)
    print(sentence)
    print(new_sentence)



    # par_file = 'C:\\Users\\MICHALD2\\projects\\Translator\\Michal\\sentences\\inss_all_pages.parquet'
    # df_inss = pd.read_parquet(par_file, engine='pyarrow')

    # counter = 0
    # modified_all = set()
    # for i, row in df_inss.iterrows():
    #     sentence = row['HE_sentences']
    #     new_sentence, modified = add_acronym_meaning(sentence, d_meaning, acronyms_container, sentence_splitter)
    #     # if is_modified:
    #     if len(modified) > 0:
    #         if len(modified.intersection(modified_all)) == 0:
    #             counter += 1
    #             print('------')
    #             print(sentence)
    #             print(new_sentence)
    #         modified_all.update(modified)
    #     if counter == 20: break
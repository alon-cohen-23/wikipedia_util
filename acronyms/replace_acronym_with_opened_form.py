import pandas as pd
import regex

STANZA_SUPPORT = False
TRIE_SUPPORT = True

if STANZA_SUPPORT:
    import stanza
    # stanza.download('he')    # should be done only once! TODO: add special handing for offline
    nlp = stanza.Pipeline(lang='he', processors='tokenize', tokenize_pretokenized=False, tokenize_no_ssplit=True,
                          download_method=None)

if TRIE_SUPPORT:
    from trie import Trie

def split_sentence_simple(sentence):
    return sentence.split(' ')

def split_sentence_regex(sentence):
    pattern = '[\p{Hebrew}\"\d]+|[^\s\"]'
    tokens = regex.findall(pattern, sentence)
    return tokens

def split_sentence_stanza(sentence):
    if not STANZA_SUPPORT:
        print("stanza is not supported")
        return None
    doc = nlp(sentence)

    splitted = []
    for s in doc.sentences:
        splitted.extend([token.text for token in s.tokens])
    return splitted

def split_sentence(sentence, mode='regex'):
    if mode == 'simple':
        return split_sentence_simple(sentence)
    elif mode == 'stanza':
        return split_sentence_stanza(sentence)
    elif mode=='regex':
        return split_sentence_regex(sentence)
    else:
        return None

def detokenize_sentence_punc(splitted):
    res = ''
    punc_attached_prev_chars = '.,;:?!\')]}'
    punc_attached_next_chars = '([{'
    for i in range(len(splitted)):
        if splitted[i][0] in splitted[i][0] in punc_attached_prev_chars:
            res += splitted[i]
        elif i > 0 and splitted[i - 1][-1] in punc_attached_next_chars:
            res += splitted[i]
        else:
            res = res + ' ' + splitted[i]
    return res

def detokenize_sentence_simple(splitted):
    return ' '.join(splitted)

def detokenize_sentence(splitted, mode='punc'):
    if mode == 'simple':
        return detokenize_sentence_simple(splitted)
    elif mode == 'punc':
        return detokenize_sentence_punc(splitted)
    else:
        return None


TRIE_SUPPORT = False


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


def identify_possible_word_parts(word, last_index_indicator = None):
    prefixes = ['מ','ש','ה','ו','כ','ל','ב']
    if last_index_indicator is not None:
        last_possible_prefix_index = word.find(last_index_indicator)
    else:
        last_possible_prefix_index = len(word)
    for prefix_end_index in range(last_possible_prefix_index):
        prefix_part = word[:prefix_end_index]
        word_part = word[prefix_end_index:]
        yield word_part
        if word[prefix_end_index] not in prefixes:
            break
    return None


def search_sub_acronym(word, acronyms_container=None):
    for word_part in identify_possible_word_parts(word, '"'):
        if acronyms_container is not None:
            acronym = search_acronym_container(word_part, acronyms_container)
            if acronym:
                return acronym
    return None

def search_sub_acronym(word, acronyms_container=None):
    prefixes = ['מ','ש','ה','ו','כ','ל','ב']
    last_possible_prefix_index = word.find('"')
    for prefix_end_index in range(last_possible_prefix_index):
        prefix_part = word[:prefix_end_index]
        word_part = word[prefix_end_index:]
        if acronyms_container is not None:
            acronym = search_acronym_container(word_part, acronyms_container)
            if acronym:
                return acronym
        if word[prefix_end_index] not in prefixes:
            break
    return None


def add_acronym_meaning(sentence, d_meaning, acronyms_sorted):
    modified_acronyms = set()

    splitted = split_sentence(sentence, mode='regex')
    for i in range(len(splitted)):
        word = splitted[i]

        pattern = '\p{Hebrew}+\"\p{Hebrew}+'
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
        if counter == 5: break
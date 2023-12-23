from itertools import product
import regex
from acronyms_utils import *
from sentence_splitter import split_sentence, detokenize_sentence


def get_first_n_grams(tokens, n):
    n_words_found = 0
    words = []
    for token in tokens:
        word = get_word(token)
        if word is not None:
            n_words_found += 1
            words.append(word)
        if n_words_found == n:
            return words
    return None


def get_n_grams_after(tokens, n):
    res = get_first_n_grams(tokens, n)
    # print(f'after: {res}')
    return res

def get_n_grams_before(tokens, n):
    reversed_tokens = tokens[::-1]
    reversed_words = get_first_n_grams(reversed_tokens, n)
    res = reversed_words[::-1]
    # print(f'before: {res}')
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

def find_opened_for_this_acronym(tokens, i):
    acronym = tokens[i]
    # print(acronym)
    acronym_with_possible_prefixes = identify_possible_word_parts(acronym, last_index_indicator='"')
    for ac_part in acronym_with_possible_prefixes:
        acronym_clean = clean_acronym(ac_part)
        acronyms_versions = create_partitions(acronym_clean)
        for av in acronyms_versions:
            acronym_len = len(av)
            n_grams_to_scan = []
            if i < len(tokens) - 1:
                # TODO: add support for words before
                words_after = get_n_grams_after(tokens[i + 1:], acronym_len)
                if words_after:
                    n_grams_to_scan.append(words_after)
            if i > 0:
                words_before = get_n_grams_before(tokens[:i], acronym_len)
                if words_before:
                    n_grams_to_scan.append(words_before)
            for to_scan in n_grams_to_scan:
                words_parts = []
                for w in to_scan:
                    words_parts.append([x for x in identify_possible_word_parts(w)])
                for words_parts_combination in product(*words_parts):
                    # print(words_parts_combination)
                    opened_form = is_possible_opened_form(av, words_parts_combination)
                    if opened_form:
                        print(acronym, words_parts_combination)
                        return acronym, words_parts_combination


def create_local_acronym_info(document):
    tokens = split_sentence(document, mode='regex')
    res = []
    for i in range(len(tokens)):
        if '"' in tokens[i]:
            # TODO: add support for acronyms with prefixes and suffixes
            acronym, words_parts_combination = find_opened_for_this_acronym(tokens, i)
            res.append((acronym, words_parts_combination))
    return res


if __name__=='__main__':
    document = 'צה"ל (צבא ההגנה לישראל) ינצח'
    # document = 'האסטרטגיה הישראלית בזירה זו בשנים האחרונות זכתה לכותרת "המערכה שבין המלחמות" (מב"מ).'
    document = 'הרמטכ"ל (ראש המטה הכללי) הורה לעשות את זה'
# document = 'צבא ההגנה לישראל (צה"ל) ינצח'
    res = create_local_acronym_info(document)
    print(res)
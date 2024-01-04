import regex
from regex_patterns import *

def is_acronym(word, search_type):
    pattern = ACRONYM_REGEX_PATTERN
    if search_type=='match':
        return regex.match(pattern, word)
    elif search_type=='search':
        return regex.search(pattern, word)

def get_word_by_regex(word):
    # if is_dicta:
    #     word_token = token[-1]
    # else: word_token = token
    # word_pattern = '[\p{Hebrew}\"\d]+'
    word_pattern = WORD_REGEX_PATTERN
    res = regex.search(word_pattern, word)
    return res.group(0) if res else None

def clean_acronym(acronym):
    return acronym.replace('"', '')

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
#
# def identify_possible_word_parts(word, sentence_splitter, last_index_indicator = None, check_for_prefixes=False):
#     if not check_for_prefixes:
#         yield sentence_splitter.get_base_word(word)
#         return
#     else:
#         whole_word = sentence_splitter.get_whole_word(word)
#     prefixes = ['מ','ש','ה','ו','כ','ל','ב']
#     if last_index_indicator is not None:
#         last_possible_prefix_index = whole_word.find(last_index_indicator)
#     else:
#         last_possible_prefix_index = len(whole_word)
#     for prefix_end_index in range(last_possible_prefix_index):
#         prefix_part = whole_word[:prefix_end_index]
#         word_part = whole_word[prefix_end_index:]
#         yield word_part
#         if whole_word[prefix_end_index] not in prefixes:
#             break
#     return None

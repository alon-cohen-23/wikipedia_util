import regex

ACRONYM_REGEX_PATTERN = '\p{Hebrew}+\"\p{Hebrew}+'
WORD_REGEX_PATTERN = ACRONYM_REGEX_PATTERN + '|[\p{Hebrew}\d]+|[\w]+'

def get_word(token):
    # word_pattern = '[\p{Hebrew}\"\d]+'
    word_pattern = WORD_REGEX_PATTERN
    res = regex.search(word_pattern, token)
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

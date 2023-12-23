import regex
from acronyms_utils import WORD_REGEX_PATTERN
TOKEN_REGEX_PATTERN = WORD_REGEX_PATTERN + '|[^\s]'


STANZA_SUPPORT = False
TRIE_SUPPORT = True

if STANZA_SUPPORT:
    import stanza
    # stanza.download('he')    # should be done only once! TODO: add special handing for offline
    nlp = stanza.Pipeline(lang='he', processors='tokenize', tokenize_pretokenized=False, tokenize_no_ssplit=True,
                          download_method=None)


def split_sentence_simple(sentence):
    return sentence.split(' ')

def split_sentence_regex(sentence):
    # pattern = '[\p{Hebrew}\"\d]+|[^\s\"]'
    pattern = TOKEN_REGEX_PATTERN #ACRONYM_REGEX_PATTERN + '|[\p{Hebrew}\d]+|[^\s]'
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

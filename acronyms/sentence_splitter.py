import regex
from regex_patterns  import WORD_REGEX_PATTERN
TOKEN_REGEX_PATTERN = WORD_REGEX_PATTERN + '|[^\s]'

SPLIT_MODE = 'dicta'
DETOKENIZE_MODE = 'dicta'

STANZA_SUPPORT = False
TRIE_SUPPORT = True
DICTA_SUPPORT = True

class sentenceDefaultSplitter():
    def __init__(self):
        pass
    def split_sentence(self, sentence):
        pattern = TOKEN_REGEX_PATTERN
        tokens = regex.findall(pattern, sentence)
        return tokens
    def detokenize_sentence(self, splitted):
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

    def get_base_word(self, token):
        return token
    def get_whole_word(self, token):
        return token

    def identify_possible_word_parts(self, word, last_index_indicator=None, check_for_prefixes=False):
        if not check_for_prefixes:
            yield sentence_splitter.get_base_word(word)
            return
        else:
            whole_word = self.get_whole_word(word)
        prefixes = ['מ', 'ש', 'ה', 'ו', 'כ', 'ל', 'ב']
        if last_index_indicator is not None:
            last_possible_prefix_index = whole_word.find(last_index_indicator)
        else:
            last_possible_prefix_index = len(whole_word)
        for prefix_end_index in range(last_possible_prefix_index):
            prefix_part = whole_word[:prefix_end_index]
            word_part = whole_word[prefix_end_index:]
            yield word_part
            if whole_word[prefix_end_index] not in prefixes:
                break
        return None

class sentenceSplitterDicta(sentenceDefaultSplitter):
    def __init__(self):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('dicta-il/dictabert-seg', local_files_only=True)
        self.model = AutoModel.from_pretrained('dicta-il/dictabert-seg', local_files_only=True, trust_remote_code=True)
        self.model.eval()

    def _is_splitted(self, token):
        return len(token)>1

    def split_sentence(self, sentence):
        splitted = self.model.predict([sentence], self.tokenizer)[0][1:-1]
        splitted_simple = super().split_sentence(sentence)
        # handle edge case: when the sentence is only one word long, the predictor sometimes return it in a different format
        if len(splitted_simple)==1 and len(splitted)>1:
            print('edge case', sentence)
            return [[sentence]]
        for i in range(len(splitted)):
            token = splitted[i]
            if self._is_splitted(token):
                whole_token = self.get_whole_word(token)
                # special handing for short acronyms (len==3)
                if len(whole_token)<=3:
                    splitted[i] = [whole_token]
        return splitted
    def detokenize_sentence(self, splitted):
        words = []
        for s in splitted:
            words.append(self.get_whole_word(s))
        return super().detokenize_sentence(words)
    def get_base_word(self, token):
        return token[-1]
    def get_whole_word(self, token):
        return ''.join(token)

    def identify_possible_word_parts(self, word, last_index_indicator=None, check_for_prefixes=False):
        base_word = self.get_base_word(word)
        yield base_word
        if check_for_prefixes is False:
            return
        elif len(word)==1:
            return
        else:
            for i in range(len(word[0])-1,-1,-1):
                res = word[0][i:]+base_word
                yield res
# if SPLIT_MODE=='stanza':
#     import stanza
#     # stanza.download('he')    # should be done only once! TODO: add special handing for offline
#     nlp = stanza.Pipeline(lang='he', processors='tokenize', tokenize_pretokenized=False, tokenize_no_ssplit=True,
#                           download_method=None)
#
#
# def split_sentence_stanza(sentence):
#     # if not STANZA_SUPPORT:
#     #     print("stanza is not supported")
#     #     return None
#     doc = nlp(sentence)
#
#     splitted = []
#     for s in doc.sentences:
#         splitted.extend([token.text for token in s.tokens])
#     return splitted


if __name__=='__main__':
    sentence = 'בשנת 1948 השלים אפרים קישון את לימודיו בפיסול מתכת ובתולדות האמנות והחל לפרסם מאמרים הומוריסטיים'
    # sentence = 'וכשהרמטכ"ל'
    sentence = 'המל"טים המריאו בחשיכה'
    sentence = 'המל"טים'
    print(sentence)
    # sentence_splitter = sentenceDefaultSplitter()
    sentence_splitter = sentenceSplitterDicta()
    splitted = sentence_splitter.split_sentence(sentence)
    print(splitted)
    new_sentence = sentence_splitter.detokenize_sentence(splitted)
    print(new_sentence)
    print(sentence==new_sentence)



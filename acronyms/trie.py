class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_word = False


class Trie:
    def __init__(self):
        self.root = TrieNode()
        self.n_words = 0

    def __len__(self):
        return self.n_words

    def insert(self, word: str) -> None:
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                curr.children[ch] = TrieNode()
            curr = curr.children[ch]
        curr.is_word = True
        self.n_words += 1

    def insert_batch(self, l_words: list) -> None:
        for word in l_words:
            self.insert(word)

    def search(self, word: str) -> bool:
        curr = self.root
        for ch in word:
            if ch not in curr.children:
                return False
            curr = curr.children[ch]
        return curr.is_word

    def search_prefix(self, word: str) -> bool:
        '''
        check if the words in the trie are prefixes of the given word
        '''
        curr = self.root
        if len(word) == 0:
            return word
        res = ''
        for ch in word:
            # print(ch, end='_')
            if curr.is_word:
                return res
            if ch not in curr.children:
                # print('wrong')
                return None
            curr = curr.children[ch]
            res += ch
            # print('found')
        if curr.is_word:
            return res
        else:  # word is shorter than acronym
            return None


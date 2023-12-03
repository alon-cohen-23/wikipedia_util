from typing import List, Tuple


import nltk
from nltk.tokenize import NLTKWordTokenizer

from string import punctuation


class LineInfo:
    def __init__(self, line):
        self.line = line
        self.words = nltk.word_tokenize(line)
        self.word_spans: List[Tuple[int, int]] = list(NLTKWordTokenizer().span_tokenize(line))

    def _strip(self, start: int, end: int) -> Tuple[int, int]:
        """
        Strip all punctuation before and after the string. Notice the input indices match Python slices
        """
        line_entity_start = start
        line_entity_end = end

        while True:
            s = self.line[line_entity_start]
            if s not in punctuation:
                break
            line_entity_start += 1

        # since the input indices are a Python slice, the end is actually +1
        while True:
            e = self.line[line_entity_end-1]
            if e not in punctuation:
                break
            line_entity_end -= 1

        # check legality of new indices
        if line_entity_start >= line_entity_end or line_entity_start > len(self.line) or line_entity_end <= 0:
            return -1, -1
        # check if anything was stripped
        if line_entity_start == start and line_entity_end == end:
            return -1, -1
        return line_entity_start, line_entity_end

    def strip_punctuation(self, word_start: int, word_end: int):
        # strip punctuation from start in word_spans
        line_entity_start, line_entity_end = self._strip(self.word_spans[word_start][0],
                                                         self.word_spans[word_end - 1][1])

        # either failed to remove punctuation or no punctuation
        if line_entity_start == -1 or line_entity_end == -1:
            return

        name_before_strip = self.get_name(word_start, word_end)
        # update start
        if self.word_spans[word_start][0] != line_entity_start:
            start_word_span = list(self.word_spans[word_start])
            start_word_span[0] = line_entity_start
            self.word_spans[word_start] = tuple(start_word_span)

        # update end
        if self.word_spans[word_end - 1][1] != line_entity_end:
            end_word_span = list(self.word_spans[word_end - 1])
            end_word_span[1] = line_entity_end
            self.word_spans[word_end - 1] = tuple(end_word_span)

        print(f"Stripped {name_before_strip} to: {self.get_name(word_start, word_end)}")

    def start(self, word_start_index) -> int:
        return self.word_spans[word_start_index][0]

    def end(self, word_end_index) -> int:
        return self.word_spans[word_end_index - 1][1]

    def get_name(self, word_start_index, word_end_index) -> str:
        return self.line[self.start(word_start_index):self.end(word_end_index)]


if __name__ == '__main__':
    ln = r"'word1 word2- \word3 word4"

    line_info = LineInfo(ln)
    # expected: ["'word1", 'word2-', '\\word3', 'word4']
    print(line_info.words)
